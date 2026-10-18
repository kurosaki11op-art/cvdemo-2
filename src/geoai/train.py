"""Train the Siamese U-Net on xBD-style tiles and save weights (data/siamese.pt).

Tile layout (xBD from https://xview2.org, or your own): <root>/{pre,post}/<id>.npy  float32 (4,H,W) R,G,B,NIR in [0,1]
                                                         <root>/mask/<id>.npy      uint8 (H,W) values 0-4
  python -m src.geoai.train --root data/xbd --epochs 20 --out data/siamese.pt
  python -m src.geoai.train --smoke     # random-tensor shape/loss/step check, no data needed
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.geoai.siamese_unet import HybridFocalDiceLoss, SiameseUNet


class TileSet(Dataset):
    def __init__(self, root):
        self.root = Path(root); self.ids = sorted(p.stem for p in (self.root / "mask").glob("*.npy"))
        if not self.ids: raise SystemExit(f"No tiles under {root}/mask")

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        r = lambda d: torch.from_numpy(np.load(self.root / d / f"{self.ids[i]}.npy"))
        return r("pre").float(), r("post").float(), r("mask").long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="data/siamese.pt"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = SiameseUNet().to(dev); loss_fn = HybridFocalDiceLoss([.1, 1, 2, 2, 2]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4)
    if a.smoke:
        pre, post = torch.rand(2, 4, 128, 128, device=dev), torch.rand(2, 4, 128, 128, device=dev)
        y = torch.randint(0, 5, (2, 128, 128), device=dev); out = net(pre, post)
        assert out.shape == (2, 5, 128, 128); l = loss_fn(out, y); l.backward(); opt.step()
        print("smoke OK, loss", float(l)); return
    dl = DataLoader(TileSet(a.root), a.bs, shuffle=True, num_workers=2)
    for ep in range(a.epochs):
        tot = 0.0
        for pre, post, y in dl:
            pre, post, y = pre.to(dev), post.to(dev), y.to(dev)
            opt.zero_grad(); l = loss_fn(net(pre, post), y); l.backward(); opt.step(); tot += float(l)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), a.out)


if __name__ == "__main__":
    main()
