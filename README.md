# Retail Footfall Counter & 2D Spatial Heatmap

YOLOv8 + ByteTrack people tracking, bi-directional tripwire counting, homography mapping to a
top-down floor plan, and a live Gaussian dwell-time heatmap in one 1280x480 OpenCV window
(left: 640x480 live feed, right: 640x480 analytics = floor-plan heatmap + KPI column).

## Setup
```
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python footfall_heatmap.py --source store.mp4 --loop        # video file
python footfall_heatmap.py --source 0 --webcam-demo         # desk test: lean right = IN, left = OUT
python footfall_heatmap.py --source store.mp4 --pick-points # click to print calibration coordinates
```
Keys: `q` quit, `r` reset counts and heatmap.

## Calibrate for your footage
Edit the constants at the top of `footfall_heatmap.py` (coordinates are in the 640x480 frame):
`FLOOR_SRC_POINTS` (4 floor corners: TL, TR, BR, BL), `FLOOR_DST_POINTS` (grid corners) and
`TRIPWIRE` (or pass `--line X1,Y1,X2,Y2`; `--in-direction right_to_left` swaps IN/OUT).

## Useful flags
`--conf` detection confidence, `--min-frames` frames before a track counts, `--no-twin-filter`
(keep look-alike boxes, e.g. two people in identical clothes), `--no-reid`, `--anchor center|feet`,
`--scale` window size.

Current Occupancy = Entries - Exits, never below zero.

## NEXUS Vision Node (alternative single-class build)
`nexus_vision_node.py` is a standalone `NexusVisionNode` class with 640x640 panes (1280x640
dashboard), KPI strip on the heatmap pane, homography corners drawn on the camera pane and a
per-frame 0.99 heatmap decay. Calibration constants are at the top of the file.
```
python nexus_vision_node.py --source 0 --anchor center --line 320,540,320,100
python nexus_vision_node.py --source meeting_room.mp4 --loop
```
