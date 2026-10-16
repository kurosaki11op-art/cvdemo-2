# Store analytics: web analytics for a physical shop

The camera turns a shop into something you can analyse the way you analyse a website.

| Website | Physical store (NEXUS) |
|---|---|
| Page | **Zone**: a named floor area (Entrance, Display A, Checkout...) drawn on the top-down map |
| Page views | **Zone stops**: standing in a zone for at least `min_zone_dwell_s` (default 3 s) |
| Unique visitors | **Visitors**: one anonymous track from first to last sighting |
| Time on page | **Dwell time** per stop |
| Bounce rate | Share of visitors who never stopped at any zone beyond the entrance |
| User flow / paths | Zone-to-zone transitions and the most common routes |
| Conversion funnel | Visited the store -> stopped at a zone -> reached checkout (or your own ordered steps) |
| Traffic by hour | Arrivals per hour |
| Heatmap (scroll/click map) | Floor heatmap of where people stood, with the zones outlined |

## Questions it answers
- **Which zones are visited most and least?** Ranked by share of visitors, with average stop time and share of total dwell. Zones nobody visited are listed too (dead zones).
- **Where do people spend time?** Dwell per stop and share of all zone time.
- **What route do they take?** Top paths and the transition table.
- **Who leaves without engaging?** Bounce rate.
- **How many reach checkout?** Funnel and drop-off step.
- **When is it busy?** Arrivals by hour, per day or per week.

Findings are generated from the numbers (never invented) and say how small the sample is when it is small.

## Set it up
```bash
nexus init-config --out configs/nexus.yaml --source rtsp://...      # includes sample zones
nexus calibrate --config configs/nexus.yaml --source rtsp://...     # floor corners + door line
nexus zones --config configs/nexus.yaml --source rtsp://... --names "Entrance,Display A,Display B,Checkout"
nexus run --config configs/nexus.yaml                                # collect
nexus dashboard --config configs/nexus.yaml                          # http://127.0.0.1:8080
nexus report --config configs/nexus.yaml --days 7                    # text report for slides
```
Zones are drawn on the top-down floor view, so they stay correct whatever the camera angle. Names containing
"entrance/door" become `entrance` zones (excluded from rankings and bounce), names containing "checkout/till/cash"
become `checkout` zones (used by the funnel); everything else is an `area`.

## Try it before you have a camera
```bash
nexus seed-demo --config configs/nexus.yaml --days 7      # generated, clearly labelled DEMO data
nexus dashboard --config configs/nexus.yaml
```
Demo data is random but plausible (busy and dead zones, lunch and evening peaks, 14% bounce). The dashboard shows a
banner and `nexus report` states it. `seed-demo` refuses to touch a database that holds real data unless `--force`.

## How a visit becomes numbers
1. The tracker gives every confirmed person an id. Their feet position is mapped to the floor map (homography).
2. `SessionTracker` assigns the position to a zone. A change of zone must hold for `switch_hold_s` (default 0.6 s), so
   standing on a border does not create many fake stops.
3. Time in a zone shorter than `min_zone_dwell_s` is dropped (walking past).
   *Stitching*: if the tracker changes a person's id mid-visit, the new id continues the old session when it appears
   within `stitch_s` (default 3 s) and a short distance from where the old one vanished (not at the entrance).
4. A person unseen for `session_timeout_s` (default 6 s) has left: the session is written to the database.

Stored per visit: start/end time, whether they crossed the door line, and the ordered zone stops with dwell times.
No images, faces, ids or appearance data (see PRIVACY.md).

## Accuracy caveats
- Visitor and zone numbers are only as good as detection and tracking. A person whose track breaks and restarts can
  appear as two visitors (the identity filter repairs many, not all). Measure on your footage with `nexus evaluate`.
- Zone stops depend on calibration. Check the heatmap against the real floor: hot spots should sit where people stand.
- Rankings from few visitors are noisy; the findings say so below 30 visitors.
- "Visitors" counts everyone the camera tracks (including staff). Position the camera or define the zones to exclude
  staff-only areas if that matters.
