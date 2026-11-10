"""ProjectQuack CCTV track: traffic speed analytics on Long Island Expressway cameras.

Modules (read them in this order if you are new):

- ``registry``       load the committed camera manifest (data/cameras_lie.json)
- ``geometry``       image pixels <-> road-plane metres (homography + calibration files)
- ``speed``          track points -> speed estimate with uncertainty (pure, tested)
- ``camera_motion``  detect that a PTZ camera moved since calibration
- ``ingest``         record HLS clips with frame timestamps
- ``detect_track``   YOLO + ByteTrack over a clip (needs requirements-detect.txt)
- ``store``          SQLite event store
- ``pipeline``       CLI: clip -> tracks -> speeds -> SQLite
- ``web.app``        FastAPI site on :8010 (map, events, stats, calibration helper)

Speeds produced here are estimates for traffic analytics. They are not
enforcement evidence and nothing here identifies vehicles or people.
"""

__version__ = "0.1.0"
