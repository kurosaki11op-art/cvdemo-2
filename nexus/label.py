"""Ground-truth labelling tool: watch a clip and press a key at every real crossing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2

from .evaluate import Event, load_truth_csv, save_truth_csv


def run_labeler(video: Union[str, Path], out_csv: Union[str, Path], speed: float = 1.0) -> int:
    """Play a clip; i = log IN, o = log OUT, u = undo, space = pause, ,/. = step, q = save+quit.

    Existing rows in `out_csv` are kept, so labelling can be done over several sessions.
    Returns the number of labelled events.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    events: List[Event] = load_truth_csv(out_csv) if Path(out_csv).is_file() else []
    delay = max(1, int(1000 / (fps * speed)))
    title = "NEXUS labeler  [i]=IN [o]=OUT [u]=undo [space]=pause [, .]=step [q]=save+quit"
    paused, index, frame = False, 0, None
    try:
        while True:
            if not paused or frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
            t = index / fps
            shown = frame.copy()
            n_in = sum(1 for _, d in events if d == "IN")
            n_out = len(events) - n_in
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(shown, f"t={t:6.2f}s  labelled IN={n_in} OUT={n_out}" + ("  [PAUSED]" if paused else ""),
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, shown)
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("i"), ord("o")):
                events.append((t, "IN" if key == ord("i") else "OUT"))
            elif key == ord("u") and events:
                events.pop()
            elif paused and key == ord("."):
                frame = None                     # advance one frame
            elif paused and key == ord(",") and index > 1:
                index -= 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                frame = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_truth_csv(events, out_csv)
    return len(events)
