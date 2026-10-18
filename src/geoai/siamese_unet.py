"""Shared-weight Siamese ResNet-34 U-Net for bi-temporal xBD damage segmentation (5 classes)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

NUM_CLASSES = 5  # 0 background, 1 undamaged, 2 minor, 3 major, 4 destroyed


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class Encoder(nn.Module):
    def __init__(self, in_ch=4, pretrained=False):
        super().__init__()
        r = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)  # R,G,B,NIR
        r.conv1.weight.data[:, :3] = w
        r.conv1.weight.data[:, 3:] = w.mean(1, keepdim=True)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool, self.l1, self.l2, self.l3, self.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        s0 = self.stem(x)                 # 64  @ /2
        s1 = self.l1(self.pool(s0))       # 64  @ /4
        s2 = self.l2(s1)                  # 128 @ /8
        s3 = self.l3(s2)                  # 256 @ /16
        s4 = self.l4(s3)                  # 512 @ /32
        return [s0, s1, s2, s3, s4]


class SiameseUNet(nn.Module):
    CH = [64, 64, 128, 256, 512]

    def __init__(self, in_ch=4, n_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.enc = Encoder(in_ch, pretrained)  # shared weights
        fused = [c * 3 for c in self.CH]       # [F_pre, F_post, |F_post - F_pre|]
        self.dec4 = _block(fused[4] + fused[3], 256)
        self.dec3 = _block(256 + fused[2], 128)
        self.dec2 = _block(128 + fused[1], 64)
        self.dec1 = _block(64 + fused[0], 64)
        self.head = nn.Conv2d(64, n_classes, 1)

    def forward(self, pre, post):
        fp, fq = self.enc(pre), self.enc(post)
        f = [torch.cat([a, b, (b - a).abs()], 1) for a, b in zip(fp, fq)]
        up = lambda x, ref: F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec4(torch.cat([up(f[4], f[3]), f[3]], 1))
        x = self.dec3(torch.cat([up(x, f[2]), f[2]], 1))
        x = self.dec2(torch.cat([up(x, f[1]), f[1]], 1))
        x = self.dec1(torch.cat([up(x, f[0]), f[0]], 1))
        return F.interpolate(self.head(x), size=pre.shape[-2:], mode="bilinear", align_corners=False)


class HybridFocalDiceLoss(nn.Module):
    """L = 0.5 * Focal + 0.5 * Dice (Dice over damage classes 1..C-1)."""

    def __init__(self, alpha=None, gamma=2.0, eps=1e-6):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha if alpha is not None else [1.0] * NUM_CLASSES))
        self.gamma, self.eps = gamma, eps

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1)
        p = logp.exp()
        y = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        a = self.alpha.view(1, -1, 1, 1)
        focal = -(a * (1 - p) ** self.gamma * y * logp).sum(1).mean()
        inter = (p * y).sum((0, 2, 3))[1:]
        denom = p.sum((0, 2, 3))[1:] + y.sum((0, 2, 3))[1:]
        dice = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()
        return 0.5 * focal + 0.5 * dice


def damage_to_polygons(mask, transform, crs="EPSG:4326"):
    """Raster damage mask -> GeoDataFrame of polygons (class>0) via rasterio.features.shapes."""
    import geopandas as gpd
    import numpy as np
    from rasterio.features import shapes
    from shapely.geometry import shape
    m = mask.astype("uint8")
    rows = [(shape(g), int(v)) for g, v in shapes(m, mask=m > 0, transform=transform)]
    return gpd.GeoDataFrame({"damage": [r[1] for r in rows]}, geometry=[r[0] for r in rows], crs=crs)
