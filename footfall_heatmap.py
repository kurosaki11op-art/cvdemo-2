#!/usr/bin/env python3
"""Backward-compatible launcher: same flags as before, now served by the `nexus` package.

    python footfall_heatmap.py --source 0 --webcam-demo --conf 0.6
    python footfall_heatmap.py --source store.mp4 --loop
"""

import sys

from nexus.cli import main

if __name__ == "__main__":
    sys.exit(main(["demo"] + sys.argv[1:]))
