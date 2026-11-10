# Source recon: where the LIE camera feeds come from

Recon done on 2026-11-10 (all "accessed" dates below are that day). The question was
whether we can measure per-vehicle speeds on the Long Island Expressway from public
cameras, and from which source we are allowed to do it.

**Short answer.** Yes, for many of them. opencctv.org is an aggregator. The LIE cameras it
lists come from NYSDOT through 511NY, as real HLS video at about 15 frames per second. We
use opencctv.org only to discover cameras and take the camera list and video from the
official NYSDOT/511NY source. Queens also has NYC DOT cameras, but those are still images
refreshed about every 2 seconds. That is too slow to follow a car, and NYC DOT asks for a
data-sharing agreement anyway.

## 1. opencctv.org

| Item | Finding |
|---|---|
| What it is | An aggregator ("140,000+ live cameras", "460+ registered public sources"), run by Impactful Technology LLC. It operates no cameras of its own. [about](https://opencctv.org/about) |
| How it works | A single-page app (Leaflet + markercluster) backed by a JSON API under `/api/` (e.g. `/api/cameras/markers`). Each camera also has a server-rendered page with JSON-LD that gives the name, coordinates, operator and the upstream stream URL. |
| robots.txt | `Allow: /cameras/ /livetraffic/ /satellites /open-cameras /about`, `Disallow: /api/ /admin`. [robots.txt](https://opencctv.org/robots.txt) |
| Terms | No terms-of-use page found; only a [privacy policy](https://opencctv.org/privacy) (its "Pro" tier records cameras with 24 h retention and no face/plate recognition). |
| LIE coverage | [I-495 / New York listing](https://opencctv.org/cameras/united-states/new-york/roadway/i-495): 85 cameras on 6 pages. Example camera page: "Camera operated by NYSDOT / 511NY", feed type M3U8, stream `https://s52.nysdot.skyvdn.com/rtplive/R10_017/playlist.m3u8`. |

**What we did.** We made 7 requests to robots-allowed pages: the listing (6 pages) and one
camera page. We never called `/api/`. 82 of the 85 listed cameras are cross-linked in our
manifest (`opencctv_url`). We did not use opencctv.org for anything else, because the
upstream source is official and has written terms.

## 2. NYSDOT / 511NY (the official source)

| Item | Finding |
|---|---|
| Public map | [511ny.org](https://511ny.org) now runs on the Castle Rock "CARS 511x" platform. The map's camera layer loads `https://api-511x-nysdot.carsprogram.org/cameras/map-features`. That returned 2,901 cameras statewide: 1,825 NYSDOT and 1,076 NYC DOT, with 1,746 video views and 1,158 still-image views. |
| robots.txt | Neither 511ny.org nor the API host publishes a robots.txt; both return the app shell or a health JSON. |
| Terms | [Developer Access Agreement](https://511ny.org/help/24), last updated 2026-09-18. Access is free, but data feeds (traffic events, **camera inventory (XML)**, winter roads, signs) require a [request form](https://forms.gle/WPYCgnKgQ727muda9) that is reviewed weekly. Data may be redistributed and enhanced on these conditions: keep its meaning, show when it was last updated, follow the "powered by 511NY" logo rules (no implied endorsement), and pass on the NYSDOT disclaimer. |
| Video hosting | NYSDOT Wowza servers: `https://s{7,9,51,52,53}.nysdot.skyvdn.com/rtplive/<R10\|R11>_<nnn>/playlist.m3u8`. R10 is NYSDOT Region 10 (Long Island) and R11 is Region 11 (NYC). The master playlist has one variant (e.g. `RESOLUTION=640x480, CODECS=avc1.4d401e, BANDWIDTH≈215 kbps`). Chunklists are 3 × 5.005 s segments with `TARGETDURATION:6`. CORS is `*`; there is no robots.txt. |

**What we did.** We took one snapshot of the map's camera layer in a normal browser
session, the same request every visitor's browser makes. It was filtered to the LIE
corridor and saved as `data/raw/511ny_cameras_lie_corridor_2026-11-10.json`, from which
`python -m cctv.recon.build_manifest` builds `data/cameras_lie.json` offline. We then made
about 20 short recordings (15–60 s each), one stream per camera, as a single viewer would.

**Before anything continuous**, Eddie should file the 511NY developer request. It covers
the camera inventory feed. We should also ask NYSDOT about continuous video pulls:
108 streams × ~240 kbps is about 26 Mbit/s around the clock, or ~8 TB a month, from
their servers.

## 3. NYC DOT (Queens only)

| Item | Finding |
|---|---|
| Feed | Still JPEGs at `https://nyctmc.org/api/cameras/<uuid>/image`, 720×480 or 352×240, `Cache-Control: no-store`. No robots.txt (404). |
| Refresh | Measured by polling once a second for 20 s: a new image every **~1.9 s** (0.5 fps). |
| Terms | The NYC DOT [data feeds page](https://www.nyc.gov/html/dot/html/about/datafeeds.shtml) says developers who want the camera feed must contact `tmcdot@dot.nyc.gov` and sign a data-sharing agreement, and must show a set disclaimer. In 2024 NYC DOT sent a cease-and-desist to the "Traffic Cam Photobooth" site ([404 Media](https://404media.co/traffic-cam-photobooth-cease-and-desist), secondary source). |
| Decision | They are in the manifest for completeness (21 cameras), but not used, and the site does not show their images. At 0.5 fps they could support counts and density at best, never speeds. |

## 4. LIE coverage (data/cameras_lie.json)

The manifest covers the Queens-Midtown Tunnel to Riverhead. The easternmost camera is at
Exit 72 (NY 25); none was found at Exit 73.

| | Count |
|---|---|
| Cameras along I-495 | **129** |
| NYSDOT HLS video / NYC DOT stills | 108 / 21 |
| Mainline / ramp or connector / cross route at interchange / service road / rest-area view | 102 / 7 / 10 / 4 / 6 |
| Queens / Nassau / Suffolk (approximate county cut-offs) | 64 / 24 / 41 |
| Mainline video cameras | 85 (55 in Region 10, 30 in Region 11) |
| Excluded: indoor LIE Welcome Center cameras (kitchen, store, hallways; people, not traffic) | 24 |

Each camera has an id (the NYSDOT stream id, e.g. `R10_041`, or `NYCDOT_<uuid8>`), name,
owner, lat/lon, county, kind, exit, a direction hint where the name gives one, chainage
(km from the tunnel), feed type and URL, posted limit, opencctv.org link, measured
characteristics, and a speed-suitability verdict.

## 5. Measured feed characteristics

`python -m cctv.recon.measure_feed` recorded 20 s from each camera, one at a time, between
21:45 and 21:52 UTC (dusk). Results are in `data/feed_measurements.json`.

| Camera | Location | County | Resolution | Effective fps | Unique fps | Frame interval jitter (std / max) | Verdict |
|---|---|---|---|---|---|---|---|
| R11_127 | I-495 at Queens Midtown Tunnel | Queens | 352×240 | 15.00 | 15.0 | 0.8 / 68 ms | marginal (low res) |
| R11_137 | I-495 at 60th Street | Queens | 352×240 | 15.00 | 15.0 | 0.8 / 68 ms | marginal |
| R11_143 | I-495 at Grand Central Parkway | Queens | 352×240 | 15.00 | 15.0 | 0.8 / 68 ms | marginal |
| R11_145 | I-495 at Lawrence Street | Queens | 352×240 | 15.00 | 15.0 | 0.9 / 68 ms | marginal |
| R11_150 | I-495 at 185th Street | Queens | 352×240 | 15.00 | 15.0 | 0.8 / 68 ms | marginal |
| R11_255 | I-495 at 228 Street | Queens | 640×352 | 14.98 | 15.0 | 0.8 / 68 ms | marginal |
| R10_011 | I-495 West of New Hyde Park Rd | Nassau | 640×480 | 14.98 | 15.0 | 0.4 / 67 ms | good |
| R10_014 | I-495 at Searingtown Road | Nassau | 640×480 | 14.99 | 15.0 | 0.4 / 67 ms | good |
| R10_017 | I-495 at Glen Cove Rd | Nassau | 640×480 | 14.98 | 15.0 | 0.4 / 67 ms | good |
| R10_019 | I-495 between Exits 40 and 39 | Nassau | 640×480 | 14.99 | 15.0 | 0.4 / 67 ms | good |
| R10_027 | I-495 at Plainview Rd | Nassau | 352×240 | 14.99 | 15.0 | 0.4 / 67 ms | marginal |
| R10_036 | I-495 at Wicks Road | Suffolk | 640×480 | 14.98 | 15.0 | 0.4 / 67 ms | good |
| R10_041 | I-495 at Veterans Memorial Hwy (Exit 57) | Suffolk | 640×480 | 14.99 | 15.0 | 0.4 / 67 ms | good (calibrated) |
| R10_044 | I-495 at Patchogue-Holbrook Rd | Suffolk | 640×480 | 14.99 | 15.0 | 0.4 / 67 ms | good |
| R10_355 | LIE at Horse Block Rd | Suffolk | 640×480 | 14.99 | (dark) | 0.4 / 67 ms | re-measure in daylight |
| R10_357 | LIE at Moriches Middle Island | Suffolk | 768×432 | 14.89 | 14.8 | 7.7 / 200 ms | good, but drops frames |
| R10_363 | LIE at NY 25, Exit 72 | Suffolk | 768×432 | 14.79 | 6.6 | 9.5 / 200 ms | marginal (55% repeated frames) |
| NYCDOT_67f77766 | LIE at QMT / Pulaski Br | Queens | 352×240 | 0.50 | – | refresh ~1.9 s | snapshot only |
| NYCDOT_43e3d020 | LIE EB at 27th St | Queens | 720×480 | 0.50 | – | refresh ~1.9 s | snapshot only |

What the measurements say:

- **Frame timing is excellent on the Region 10 640×480 encoders.** Presentation timestamps
  step 66.7 ms with 0.4 ms standard deviation. There were no gaps and no repeated frames,
  and segments are published every 4.9–5.0 s of wall-clock time, i.e. in real time. So the
  stream's PTS can be trusted as the clock for speed.
- **The newer east-end cameras (R10_355–R10_363, 768×432) drop and repeat frames**, with
  gaps up to 200 ms and up to 55% repeats. PTS still say when each frame was sent, but
  repeated pictures carry stale positions. The error budget shows 50% repeats cost about
  3 mph of bias and half the tracks.
- **Queens (Region 11) streams are 352×240.** Speeds there are possible only near the
  camera, at lower precision.
- **The burned-in clocks are wrong.** Camera 017 and 011 ran about 4 minutes fast, camera
  363 about 20 minutes slow, and 357 was about right. Never use the on-screen clock for
  timing or latency.
- **Latency.** Segments are 5.005 s long, and a player joins 1–3 segments behind live. A
  live pipeline will run about 10–20 s behind real time. That is fine for analytics, but
  real-time alerts are not realistic.
- **PTZ is real.** Camera 357 was zoomed onto an incident (emergency vehicles) while we
  measured it. Camera 041 faced the other way an hour after we calibrated it. Camera 363 is labelled "(Dome)", and only one camera (R10_314) is named
  "(Fixed)". Calibrations must be checked against a reference view on every run
  (`cctv/camera_motion.py`).

**How many LIE cameras can give per-vehicle speeds?** All 108 NYSDOT cameras deliver real
video at about 15 fps, and the 85 mainline ones are the candidates.

- **Likely good:** roughly 40–45 Region 10 mainline cameras in Nassau and Suffolk, if the
  640×480 / clean-timing pattern holds for the unmeasured ones. That is 8 of 9 measured.
- **Marginal:** about 30 Queens cameras (352×240) and the 9 east-end cameras (frame
  drops).

Whether a given camera works also depends on its current view: enough visible road, lane
dashes to calibrate on, and not pointed at the horizon. That has to be checked camera by
camera, and re-checked whenever the operator moves it. Snapshot-only cameras and
uncalibrated video cameras can still show **counts, flow (vehicles/hour) and density**,
and the site shows those where no speeds exist.

## 6. Geometry and legal references

| Fact | Source |
|---|---|
| Broken lane lines: 10 ft segments, 30 ft gaps (Guidance, "should") | MUTCD 11th ed. (Dec 2023) §3A.04 ¶06 ([PDF](https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/part3.pdf)); same text in the 2009 ed. §3A.06 ([html](https://mutcd.fhwa.dot.gov/htm/2009/part3/part3a.htm)) |
| NYSDOT standard: 10 ft line / 30 ft gap, 6 in wide on freeways | NYSDOT Standard Sheet 685-01 ([PDF](https://www.dot.ny.gov/main/business-center/engineering/cadd-info/drawings/standard-sheets-us-repository/685-01_020221.pdf)) |
| 12 ft Interstate lanes | FHWA, Interstate System – Design ([page](https://www.fhwa.dot.gov/programadmin/interstate.cfm)) |
| LIE limits in Queens: mostly 50 mph; 45 mph from the tunnel plaza to about 48th St | NYC DOT posted speed limit dataset ([5mad-ntua](https://data.cityofnewyork.us/resource/5mad-ntua.json)), our reading of segment extents |
| LIE limit in Nassau/Suffolk: 55 mph (40 mph minimum) | Secondary only ([nysroads.com](https://nysroads.com/i495list.php)); consistent with the statewide 55 mph maximum, VTL §1180(b) ([statute](https://www.nysenate.gov/legislation/laws/VAT/1180)). **Needs NYSDOT confirmation.** |
| NYC school-zone speed cameras, run by the City | VTL §1180-b ([statute](https://www.nysenate.gov/legislation/laws/VAT/1180-B)) |
| Work-zone speed cameras: NYSDOT, Thruway, MTA B&T, NYS Bridge Authority | VTL §1180-e ([statute](https://www.nysenate.gov/legislation/laws/VAT/1180-E)); [program page](https://www.ny.gov/work-zone-safety-awareness/automated-work-zone-speed-enforcement-program) |
| Public speed data for validation (Queens LIE links, several updates a minute) | NYC DOT Traffic Speeds ([i4gi-tjb9](https://data.cityofnewyork.us/api/views/i4gi-tjb9.json)). Live LIE links: 84th St–108th St both directions, and westbound Little Neck Pkwy to Cross Island/Clearview. Links from the tunnel to 84th St have been stale since 2020. |
| NPMRDS travel times (INRIX, 5-minute bins) | For state DOTs and MPOs via RITIS; not public ([RITIS](https://auth.ritis.org/npmrds)) |
