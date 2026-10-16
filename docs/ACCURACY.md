# Measuring accuracy

**No accuracy figure is claimed until it is measured on your own footage.** Detectors behave very differently
by camera angle, lighting and crowd density. Use this procedure.

## 1. Camera placement (biggest factor)
- Mount high (about 2.5-4 m) and angled down at the doorway, so people are seen full-body or from above.
- The tripwire goes across the walking path just inside the door; people should cross it roughly perpendicularly.
- Avoid mirrors, glass doors facing the camera, strong back-light and screens showing people.
- Leave 1-2 m of approach space on each side of the line so tracks are established before they cross.

## 2. Record representative footage
At least 20-30 minutes across quiet and busy periods, including groups, people with bags/prams, staff and slow browsers.

## 3. Label ground truth
```bash
nexus label --source clip.mp4 --out truth/clip1.csv        # i = IN, o = OUT, space = pause, u = undo
```
Press the key at the moment each person crosses. Slow the playback with `--speed 0.5` for busy clips.

## 4. Evaluate
```bash
nexus evaluate --config configs/nexus.yaml --source clip.mp4 --truth truth/clip1.csv --report reports/clip1.json
```
Reports, per direction and overall: truth vs counted, true/false positives, false negatives, precision, recall,
count error (%), and F1. Events match when the direction is equal and the times differ by at most
`--tolerance` seconds (default 1.5). Use `--min-f1 0.9` to turn it into a regression gate.

## 5. Tune, then re-measure
Change one thing at a time and re-run on the same clips:
`detection.conf`, `detection.min_frames`, tripwire position, a larger model, `twin_filter` (disable it if staff wear identical uniforms).

## Results log (fill in with your own measurements)
| Date | Site / camera | Clip length | Truth IN/OUT | Counted IN/OUT | Precision | Recall | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Known failure modes
- Groups crossing shoulder to shoulder can merge into one detection (undercount).
- Occlusion by shelves or other people can drop and re-issue an ID (the identity filter repairs many of these, not all).
- A person standing exactly on the line is counted at most once until they move `rearm_px` away.
- Identical clothing for two people in view at once is treated as a reflection (`twin_filter`).
- Re-identification uses clothing colour within a session only; it cannot link a person across days.
