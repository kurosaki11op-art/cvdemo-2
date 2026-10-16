# Privacy

This is an engineering description and a starting template, **not legal advice**. Rules on video analytics
differ by country and sector (for example GDPR in the EU/UK); have your own counsel or DPO review a deployment.

## What the system stores
| Data | Stored? |
|---|---|
| Video or images | **No.** Frames are processed in memory and discarded |
| Faces / biometrics | **No.** No face model is used |
| Person identity or track ids | **No.** Ids exist in memory for tracking only |
| Crossing events | Yes: timestamp, camera id, `IN`/`OUT` |
| Heatmap | Yes: aggregated dwell grids (anonymous 2D density, 5-minute deltas) |
| Occupancy | Yes: a single number |

The only transient personal-looking data is a clothing-colour histogram used to keep an id stable within one
session; it lives in RAM and is never written to disk.

## Controls
- `storage.retention_days` deletes events and heatmaps automatically (hourly check). Default 90 days.
- Dashboard binds to `127.0.0.1`; `dashboard.api_token` adds a shared secret; put TLS and real authentication
  in a reverse proxy for anything beyond a trusted LAN (see DEPLOYMENT.md).
- The demo window (`nexus demo`) shows live video on screen; do not use it in public areas, and do not record the screen.
- Keep camera credentials out of version control; use environment-specific configs or secrets.

## Checklist before going live
- [ ] Legal basis identified (e.g. legitimate interest) and documented; DPIA/assessment done where required
- [ ] Signage displayed at the entrance (template below)
- [ ] Retention period decided and set
- [ ] Access to the dashboard limited to people who need it
- [ ] Camera positioned to cover the entrance only, not private areas

## Signage template
> **Visitor counting in operation.** This entrance uses a camera to count how many people enter and leave.
> No images are stored and no one is identified. Counts are kept for [N] days to help us manage staffing and
> opening hours. Contact: [name / email].
