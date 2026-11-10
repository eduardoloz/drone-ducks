# What we need: face detection on a Jetson on the drone

_Last updated 2026-11-10. Prices and specs were checked on that date (sources at the end of each
section). **(est.)** marks my own estimate, **(unverified)** marks a figure from a search snippet or
secondary source. Numbers measured on Eddie's Intel Mac are labelled **Mac**, never "Jetson"._

## 1. The answer on one page

| Decision | Recommendation | Why |
|---|---|---|
| Compute | **Jetson Orin Nano Super Developer Kit** on **JetPack 6.2.x** (not 7.x yet). If the club already owns an **original Jetson Nano**, use it for bench work and learning now; budget for the Orin for flight. | Original Nano is end-of-life, stuck on Ubuntu 18.04 / Python 3.6 / CUDA 10.2. Orin is ~5-10x faster, Python 3.10, maintained. |
| Camera | **IMX219 CSI module that explicitly says IMX219** (Arducam for Orin, 62.2 deg HFOV, ~$20) + 15-to-22-pin cable, **fixed mount tilted ~15-20 deg down**, on rubber dampers. | Supported by JetPack out of the box, 3 g. New RPi Camera Module 2 units (Nov 2026+) switched to an IMX355 sensor with no Jetson driver. A level camera can't see a face closer than ~3 m from 3 m altitude (section 5). |
| Detector | **YuNet** (OpenCV `FaceDetectorYN`, MIT, 230 KB) at 640-1280 px input; GPU later via ONNX Runtime's TensorRT provider. | Runs everywhere (laptop, Nano CPU, Orin), finds faces down to ~10-17 px, already integrated and tested here. |
| Range strategy | Face detection only works to **~5 m at 640 px / ~10-12 m at 1280 px** with a 62 deg lens. For longer range: **two stages** (person detector far away, face detector on crops up close). | Pixel maths + our own measurement (section 5). |
| Jetson to Pixhawk | **UART**: Pixhawk 6C TELEM2 to Jetson header pins 8/10 (`/dev/ttyTHS1`), 921600 baud; MAVSDK opens the serial port directly. Ground station stays on the SiK radio (TELEM1). | PX4's documented companion port; no extra router needed. |
| Control | MAVSDK Offboard `set_velocity_body` at 20 Hz from our pure controller; PX4 does all PID. Pilot with RC can always take over. | Same stack as the sim. |
| Before anyone flies | SBU EH&S approval, **Part 107** remote pilot for project flights (SBU policy), FAA registration, a **FAA-accepted Remote ID broadcast module**, TRUST for everyone, fly at the SBU R&D Park flying field. | Section 10. |

Total for the recommended onboard kit (Orin path): **about $500 (microSD) to $670 (with NVMe)** on top
of the drone (table in section 3). The drone itself (X500 V2 PX4 dev kit with Pixhawk 6C) is $609 plus battery, charger,
RC transmitter/receiver and a Remote ID module.

## 2. Which Jetson? Original Nano vs Orin Nano Super

| | Original Jetson Nano (B01, 4 GB) | Jetson Orin Nano Super Dev Kit |
|---|---|---|
| Status | **EOL.** Dev kit discontinued (EOL notice 2023-12-20); module supply to Jan 2027 | Current. Successor "Orin Nano 2" announced for H1 2027 (unverified) |
| Price now | Not sold as a kit by NVIDIA; module list $267; used kits ~$100-170 (unverified) | **$399** list (NVIDIA raised it from $249 in July 2026); often backordered |
| Last/current JetPack | **4.6.6 / L4T 32.7.6** = final release | **6.2.3 / L4T 36.5.2** (2026-08-12). JetPack 7.2.x supports Orin but has no SD image and an open UART-TX bug report, so stay on 6.2.x |
| OS / Python | Ubuntu 18.04 / **Python 3.6** | Ubuntu 22.04 / Python 3.10 |
| CUDA / TensorRT | 10.2 / 8.2.1 | 12.6 / 10.3 |
| Bundled OpenCV | 4.1.1 (GStreamer, no CUDA) - **too old for YuNet** (needs >= 4.5.4; 4.8 for the 2023 model) | 4.8.0 (GStreamer, no CUDA) - fine |
| AI throughput | 472 GFLOPS FP16 (NVIDIA spec), 128-core Maxwell GPU, 4x A57 | 67 INT8 TOPS, 1024 CUDA + 32 Tensor cores, 6x A78AE |
| Power input | 5 V 4 A barrel (J48 jumper) or 5 V 2 A micro-USB; 10 W / 5 W modes | **9-20 V** barrel (19 V supply included), max 3.5 A; 7/15/25 W / MAXN SUPER |
| Size / mass | 100 x 80 x 29 mm, ~138 g (unverified) | 103 x 90.5 x 34.8 mm, **175 g**; rated 0-35 C ambient |
| Camera ports | 2x 15-pin CSI | 2x 22-pin CSI (needs 15-to-22 cable for Pi-style cameras) |
| UART for Pixhawk | pins 8/10 -> `/dev/ttyTHS1` (disable `nvgetty`) | pins 8/10 -> `/dev/ttyTHS1` on JetPack 6 (`ttyTHS0` on JetPack 5) |
| Wi-Fi | none (add Intel AC8265 M.2 card, ~$24) | included |

**What the original Nano's limits mean in practice**

- System Python 3.6 can't run current MAVSDK (needs >= 3.8), current OpenCV wheels (4.6.0.66 was the last
  for 3.6) or this package. Two sane routes, both documented in `docs/BRINGUP_CHECKLIST.md`:
  1. **Ubuntu's `python3.8` package** in a venv + `opencv-python-headless` + **`mavsdk==4.0.6`**
     (`mavsdk` 4.1.0's ARM wheels need glibc 2.28; the Nano has 2.27). YuNet then runs on the CPU.
  2. **Ultralytics' JetPack 4 container** (`ultralytics/ultralytics:latest-jetson-jetpack4`: Python 3.8,
     TensorRT 8.2, onnxruntime-gpu 1.8) for GPU inference. dusty-nv's jetson-containers no longer
     supports JetPack 4.
- pip OpenCV has no GStreamer, so CSI capture on the Nano goes through a system GStreamer pipeline that
  serves MJPEG on localhost (`--source tcp://127.0.0.1:5000`, command in the checklist) or a USB camera.
- jetson-inference's Python bindings on JetPack 4 are built for 3.6 only (editable in its CMake file).
- Speed: YuNet on the Nano CPU is ~5.6 ms at 160x120 (opencv_zoo), so roughly **90 ms at 640x480
  (est., ~11 FPS)** and ~22 ms at 320x240 (est.). Usable for a slow follow demo at close range, not
  for range.

**Decision for Eddie:** do we already own an original Nano? If yes, start on it this week (everything
here runs on it on the CPU). Either way, the Orin is the flight computer.

## 3. Bill of materials (onboard compute + sensing)

Prices in USD, 2026-11-10. "Orin path" is the recommendation; Nano-path substitutions below it.

| Item | Pick | Key spec | Price | Source |
|---|---|---|---:|---|
| Compute | Jetson Orin Nano Super Developer Kit | 8 GB, 67 TOPS, 9-20 V in, 175 g | $399 | NVIDIA Jetson FAQ; SparkFun listing |
| Camera | Arducam IMX219 for Jetson Orin | 62.2 deg HFOV, 1080p30/720p60, ~3 g | $19.95 | SparkFun listing |
| Camera cable | 15-pin to 22-pin, 200 mm | Pi camera to Orin CSI | $2.50 | SparkFun (pin-compat. unverified) |
| Storage | SanDisk Extreme 64 GB A2 microSD (or NVMe, below) | boot + logs | $31.99 | Amazon via datacenterdisk.com |
| Storage (better) | 512 GB M.2 2230/2280 NVMe | faster boot, recordings | $103-168 | datacenterdisk.com (NAND shortage: ~2-3x 2025 prices) |
| Power | Pololu D36V50F12 (12 V, 4.5 A step-down) from the 4S battery | needs >= 13.3 V in (dips near empty, still above Orin's 9 V min) | $39.95 | pololu.com/product/4095 |
| Mount | M3 anti-vibration standoffs / damper balls + 3D-printed plate on the X500 rails | isolate camera + Jetson | ~$5-15 + print | getfpv, progressiverc (unverified) |
| Cooling | included fan (Orin) | keep under 35 C ambient spec | $0 | carrier board spec |
| Wi-Fi | included (Orin) | dev access in the field | $0 | |
| **Orin path total** | $399 + 19.95 + 2.50 + 31.99 + 39.95 + ~10 mount | | **~$503** (~$670 with NVMe) | |

Nano-path substitutions: power with a **5 V >= 5 A BEC** (Castle Creations BEC 2.0, 9 A, 2S-14S, $46.95;
or Pololu D36V50F5, 5 V 5.5 A, $39.95) into the barrel jack with the **J48 jumper fitted** (micro-USB is
limited to 2 A: brown-outs under load); a **5 V PWM fan** (Noctua NF-A4x20 5V PWM is the one Noctua says
NVIDIA recommends; NF-A4x10 ~$20, unverified); **Intel AC8265 Wi-Fi card** (Waveshare, $24.35); the
Nano takes Pi cameras with the stock 15-pin cable.

Other cameras considered (all HFOV): RPi HQ IMX477 ($55, out of stock; needs resistor R8 removed for
some Jetson drivers) with 6 mm lens ~55 deg / 16 mm lens ~22 deg (my calculation from 6.287 mm sensor
width); Logitech C920S USB 70.42 deg ($69.99, easiest to start with, heavier, rolling shutter);
Arducam OV9281 USB global shutter 70 deg mono (~$55, unverified); IMX219 "160 deg" modules are
diagonal-FOV fisheyes (~120 deg H, faces tiny). RPi Camera Module 3 (IMX708): no native Jetson support.

Drone side (for the budget, not this track): Holybro **X500 V2 PX4 Development Kit with Pixhawk 6C**
$609 (incl. M10 GPS, SiK V3 radio, PM02 power module); ARF kit $329, frame only $135; Pixhawk 6C alone
$199; SiK V3 radio $58.99. **Remote ID:** the Holybro Remote ID module ($19.99) is _not_ for DIY users
(Holybro: "do your own approval process at the FAA"); buy a broadcast module on the FAA's accepted
list, e.g. DroneBeacon db120 (~$60, unverified) or Dronetag (check uasdoc.faa.gov before buying).
An integrated alternative is Holybro's **Pixhawk Jetson Baseboard** (Pixhawk 6X + Jetson on one board,
UART/CAN/Ethernet between them, 7-21 V in): board $395.99, bundles with Orin NX 16 GB ~$2,280.

Sources: NVIDIA Jetson FAQ (developer.nvidia.com/embedded/faq); Orin Nano carrier board spec
SP-11324-001 v1.3; nvidia.com Orin Nano Super page; sparkfun.com product pages; raspberrypi.com camera
documentation (IMX355 note); pololu.com/product/4091, /4095; motionrc.com (Castle BEC 2.0);
holybro.com/products/x500-v2-kits, /px4-development-kit-x500-v2, /pixhawk-6c, /remote-id,
/pixhawk-jetson-baseboard; docs.holybro.com/radio/remote-id; robotshop.com (AC8265); adafruit.com
(HQ camera + lenses); logitech.com (C920S). All accessed 2026-11-10.

## 4. Weight and power budget

| Payload item | Orin path | Nano path |
|---|---:|---:|
| Jetson dev kit (with heatsink/fan) | 175 g | ~138 g (unverified) + ~10 g fan |
| Camera module + cable | ~5 g | ~5 g |
| Regulator / BEC | ~15-25 g (est.) | 21 g (Castle) |
| Mount plate, dampers, screws, wiring | ~60-80 g (est.) | ~60-80 g (est.) |
| **Total** | **~260-290 g** | **~235-255 g** |

- X500 V2 payload: Holybro's docs say **~1 kg with a 4S 5000 mAh battery at 70% throttle** (the store
  page says 1.5 kg without battery). Our ~0.3 kg is ~30% of that: fine.
- Flight time: Holybro quotes **~18 min hover** with no payload (5000 mAh). Rough estimate with payload
  (est.): 74 Wh x 80% usable over 18 min = ~200 W hover. +0.3 kg on a ~1.7 kg aircraft raises hover
  power ~30% (power ~ mass^1.5), and the Orin adds up to 25 W: 59 Wh / ~280 W = **~12-13 min**
  (Nano at 10 W: ~13 min). Plan test sorties of **8-10 minutes** with a landing reserve.
- Electrical: Orin up to 25 W -> ~2.1 A at 12 V (Pololu 12 V 4.5 A is fine). Nano 10 W module + carrier
  + USB camera -> 2-3 A at 5 V, peaks near 4 A, so use a >= 5 A BEC. Never run the Jetson from the
  flight controller's power module 5 V rail. Holybro recommends against powering a Jetson straight off
  the battery (voltage spikes); the Orin's 9-20 V input technically covers a 4S pack (12.8-16.8 V) but
  use the regulator.

## 5. Camera geometry: how far away can we see a face?

A face is ~0.16 m wide. With horizontal FOV `theta` and the detector seeing an image `W` px wide, a face
at distance `d` spans `px = W * 0.16 / (2 * d * tan(theta/2))` pixels. Modern detectors need roughly
**10-20 px** (YuNet's docs: "faces of pixels between around 10x10 to 300x300"; WIDER FACE ignores faces
under 10 px). `W` is what the detector sees *after resizing*. Generated by `python -m jetson.geometry`:

| Camera (HFOV) | 320 px @ 10 px | 320 px @ 20 px | 640 px @ 10 px | 640 px @ 20 px | 1280 px @ 10 px | 1280 px @ 20 px | 1920 px @ 10 px | 1920 px @ 20 px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RPi Camera v2 (IMX219, stock lens) (62.2 deg) | 4.2 m | 2.1 m | 8.5 m | 4.2 m | 17.0 m | 8.5 m | 25.5 m | 12.7 m |
| RPi HQ (IMX477) + 6 mm CS lens (55.3 deg) | 4.9 m | 2.4 m | 9.8 m | 4.9 m | 19.5 m | 9.8 m | 29.3 m | 14.7 m |
| RPi HQ (IMX477) + 16 mm C lens (22.2 deg) | 13.0 m | 6.5 m | 26.1 m | 13.0 m | 52.2 m | 26.1 m | 78.3 m | 39.1 m |
| Logitech C920 USB (78 deg diagonal, 16:9) (70.4 deg) | 3.6 m | 1.8 m | 7.3 m | 3.6 m | 14.5 m | 7.3 m | 21.8 m | 10.9 m |
| IMX219 'wide' module (~120 deg H) (120 deg) | 1.5 m | 0.7 m | 3.0 m | 1.5 m | 5.9 m | 3.0 m | 8.9 m | 4.4 m |

**Measured (Mac, synthetic scene):** YuNet on a real face photo rendered into our toy world at known
distances (62.2 deg HFOV): detected at **>= 17 px every time, 12-14 px sometimes, <= 11 px never**.
That is **~5 m at 640 px and ~12 m at 1280 px**, consistent with the table's 20 px columns.
(Real outdoor faces, blur and oblique angles will be worse.) Independent check: a published DroneFace
study found RetinaFace-MobileNet at 100% AP only within ~8.5 m horizontal distance (camera 1.5 m high).

What this means:

1. **Narrow lenses and high resolution buy range; wide lenses kill it.** A 120 deg lens at 640 px sees
   faces only to ~1.5-3 m. Stick with ~60-70 deg.
2. **Resolution costs compute** (YuNet Mac: 5 ms at 320, 11 ms at 640, 59 ms at 1280). A cheaper way
   to range: run at 640 px, and once a face is found, run the detector on a full-resolution **crop
   around the last known position** (a good first task).
3. **Two-stage for range**: a person is ~10x taller than a face is wide. A person detector needing
   ~40 px of body height sees people to **~23 m at 640 px / ~45 m at 1280 px** (62 deg lens). Find the
   person far away, approach, then switch to faces under ~8-10 m. On the Orin, YOLO11n (COCO, includes
   "person") runs at ~4.9 ms with TensorRT FP16 (Ultralytics docs).
4. **Look-down angle**: detectors are trained on roughly eye-level faces. From 5 m altitude a face 5 m
   away is seen 34 deg from above, 3 m away 49 deg. Keep the depression angle under ~30-40 deg: fly
   low (3-4 m) and stand off 4-6 m.
5. **Tilt the camera down ~15-20 deg.** Found by our closed-loop test: at 3 m altitude a *level*
   camera (vertical FOV 48.8 deg) loses a face closer than (3 - 1.6) / tan(24.4 deg) = **3.1 m**, i.e.
   exactly at the standoff distance. (`test_level_camera_cannot_see_a_close_face_from_altitude`.)
6. The HFOV in `--hfov` must match the camera *and sensor mode*: some IMX219 modes are crops (narrower
   FOV), which silently breaks the distance estimate. Calibrate `--face-width` by standing at a known
   distance (good first task).

## 6. Which face detector?

| Detector | Input | Size | WIDER FACE val AP (easy / med / hard) | Licence (code / weights) | Jetson numbers (published) |
|---|---|---|---|---|---|
| **YuNet 2023mar** (OpenCV) | any (DNN), fixed 640x640 ONNX for ORT/TRT | 232,589 B, 75,856 params | 0.884 / 0.866 / 0.750 (opencv_zoo) | MIT / MIT | at 160x120: Nano CPU 5.62 ms, Nano CUDA 10.99 ms (slower!), Orin Nano CPU 2.59 ms, CUDA 5.0-5.2 ms (opencv_zoo benchmark) |
| SCRFD-500M-KPS (InsightFace) | dynamic | 2.5 MB, 0.57 M params, 0.5 GFLOPs @VGA | 90.97 / 88.44 / 69.49 (VGA) | MIT / **non-commercial research only** | none found |
| SCRFD-2.5G-KPS | dynamic | 0.82 M params | 93.80 / 92.02 / 77.13 | same | none found |
| RetinaFace-MobileNet0.25 | original size | 1.7 MB | 90.70 / 88.16 / 73.82 | MIT / not stated | none found |
| yolov8n-face (derronqi) | 640 | ~6 MB | 94.6 / 92.3 / 79.6 | GPL-3.0 / AGPL lineage | YOLOv8n-class TRT FP16: Nano ~19 FPS, Orin Nano ~100 FPS (Q-engineering, inference only); YOLO11n 4.91 ms on Orin Nano Super (Ultralytics) |
| jetson-inference `facedetect` (TAO FaceDetect) | 736x416 | 5.5 MB | not published (proprietary set: 83.85 mAP) | MIT / NVIDIA model EULA | Nano 16 FPS (NVIDIA model card; "INT8" on a Maxwell GPU is odd) |
| MediaPipe BlazeFace | 128x128 / 192 | 0.2-1.1 MB | not published | Apache-2.0 | short-range model "within 2 m", full-range "within 5 m"; no JetPack 4 wheel |
| Haar cascade (baseline) | 24x24 window | 0.9 MB | 0.215 / 0.260 / 0.144 (3rd-party val run) | BSD-style | slow and noisy (Mac: 54 ms at 640 px with false positives) |

**Measured on the Mac** (Intel i9-9880H CPU, **not Jetson**, load average 6-14 from other jobs; full
table in `bench_results/intel_mac_cpu_2026-11-10.md`), mean ms per frame:

| backend | 320 px | 640 px | 1280 px |
|---|---:|---:|---:|
| yunet (OpenCV) | 5.0 | 11.0 | 59.1 |
| onnx:yunet (ONNX Runtime, fixed 640x640) | 10.3 | 10.1 | 10.3 |
| onnx:scrfd (SCRFD-500M) | 4.8 | 14.5 | 56.9 |
| haar | 15.2 | 53.6 (2 false +) | 156.3 (6 false +) |

Our ONNX Runtime YuNet decoder reproduces OpenCV's boxes exactly (IoU 1.000 on the test photo), so the
same MIT model can run on ORT-CPU, ORT-CUDA, ORT-TensorRT or a native TensorRT engine.

**Recommendation:** YuNet as the default everywhere (accuracy close to the others on easy/medium, best
licence, smallest, already wired in). Use SCRFD-500M for comparison (licence: research only). YOLO-face
is the most accurate on hard faces and the natural choice if we go two-stage with YOLO person
detection anyway (AGPL is fine for us; not for a product). Haar only as a teaching baseline. Expected
on hardware **(est.)**: YuNet at 640x480 on Nano CPU ~90 ms (~11 FPS); on Orin CPU ~40 ms (~25 FPS);
on Orin GPU via TensorRT a few ms. Verify with `python -m jetson.bench --save` on day one.

Sources: github.com/opencv/opencv_zoo (models/face_detection_yunet README, benchmark/README);
link.springer.com YuNet paper (s11633-023-1423-y); github.com/deepinsight/insightface (detection/scrfd,
model_zoo); github.com/biubug6/Pytorch_Retinaface; github.com/derronqi/yolov8-face;
github.com/akanametov/yolo-face; docs.ultralytics.com/guides/nvidia-jetson (v8.3.x table);
github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano; catalog.ngc.nvidia.com TAO FaceDetect card;
github.com/dusty-nv/jetson-inference (data/networks/models.json); mediapipe.readthedocs.io;
arxiv.org/abs/1511.06523 (WIDER FACE); github.com/TonyCongqianWang/cv_facedetection_eval;
hjhsu.github.io/DroneFace; arxiv.org/abs/2011.07689 (Drone LAMS). Accessed 2026-11-10.

## 7. Jetson <-> Pixhawk link

```
 Pixhawk 6C TELEM2 (JST-GH 6)          Jetson 40-pin header
   pin 2  TX (3.3 V)  -------------->  pin 10  UART RX
   pin 3  RX (3.3 V)  <--------------  pin 8   UART TX
   pin 6  GND         ---------------  pin 6   GND
   pin 1  +5 V        (leave unconnected: the Jetson has its own supply)
   pins 4/5 CTS/RTS   (unused at 921600 without flow control)
```

Both sides are 3.3 V logic (Pixhawk: "All Pixhawk serial ports operate at 3.3V"; Orin header 3.3 V is
per community sources, unverified from NVIDIA). Device: `/dev/ttyTHS1` on the original Nano (disable
`nvgetty` first) and on the Orin with JetPack 6. Add your user to `dialout`.

**PX4 parameters to set** (PX4 v1.17 source defaults in brackets; **the docs say TELEM2 is enabled by
default but the source disables it**):

| Parameter | Set to | Why |
|---|---|---|
| `MAV_1_CONFIG` | `102` (TELEM 2) | [disabled] turn on MAVLink on TELEM2 |
| `MAV_1_MODE` | `2` Onboard | [2] companion-computer message set |
| `SER_TEL2_BAUD` | `921600` | [921600] |
| `MAV_1_RATE` | `0` (half max) | [0] |

Software: MAVSDK can open the port itself, `--connect serial:///dev/ttyTHS1:921600`, which is all one
app needs. If several programs on the Jetson need MAVLink (our app + logging + QGC over Wi-Fi), run
**mavlink-router** instead and point MAVSDK at UDP:

```ini
# /etc/mavlink-router/main.conf
[UartEndpoint fc]
Device = /dev/ttyTHS1
Baud = 921600
[UdpEndpoint app]          # our MAVSDK app: --connect udpin://0.0.0.0:14540
Mode = Normal
Address = 127.0.0.1
Port = 14540
[UdpEndpoint gcs]          # QGroundControl on a laptop over Wi-Fi (dev only)
Mode = Normal
Address = <laptop IP>
Port = 14550
```

The ground station's primary link stays the **SiK telemetry radio on TELEM1** (independent of the
Jetson, so QGC keeps working if the Jetson dies). USB to the Pixhawk is for setup on the bench, not
flight. MAVSDK versions: `mavsdk` 4.1.0 on Orin/JetPack 6; **`mavsdk==4.0.6` on an original Nano**
(same native API, older glibc wheels). Our bridge also supports the classic gRPC API.

Sources: docs.px4.io (companion_computer/pixhawk_companion, peripherals/mavlink_peripherals,
flight_modes/offboard, config/safety); PX4-Autopilot `release/1.17` parameter definitions;
docs.holybro.com Pixhawk 6C pinout; jetsonhacks.com (Nano UART); forums.developer.nvidia.com/t/328934
(Orin UART on JP6); github.com/mavlink-router/mavlink-router; mavsdk.mavlink.io (connections, offboard);
pypi.org/project/mavsdk (wheel tags). Accessed 2026-11-10.

## 8. Software stack per platform

| | Laptop (dev) | Original Nano, JetPack 4.6.6 | Orin Nano, JetPack 6.2.x |
|---|---|---|---|
| Python | 3.8-3.12 (we test 3.8.20 + 3.12.8) | apt `python3.8` venv, or Ultralytics JP4 container (3.8) | system 3.10, venv `--system-site-packages` |
| OpenCV | pip `opencv-python<4.12` | pip `opencv-python-headless` (no GStreamer) | JetPack's 4.8.0 (has GStreamer) |
| Detector default | YuNet CPU | YuNet CPU | YuNet CPU -> ORT TensorRT EP |
| GPU option | - | container: TensorRT 8.2 / onnxruntime-gpu 1.8 | onnxruntime-gpu 1.23/1.24 (jetson-ai-lab index), TensorRT 10.3 + pycuda, jetson-inference |
| CSI camera | - | system GStreamer -> `tcp://` MJPEG, or USB camera | `csi://0` (OpenCV GStreamer) |
| MAVSDK | `mavsdk` 4.1 | `mavsdk==4.0.6` | `mavsdk` 4.1 |
| Requirements file | `requirements.txt` | `requirements-jetson-jp4.txt` | `requirements-jetson-jp6.txt` |

**Minimum Python for this package is 3.8** (statically checked with `vermin`; full test suite passes
on 3.8.20). That is the oldest Python with current MAVSDK wheels, and Ubuntu 18.04 ships it, so even
the original Nano runs everything without a container. Python 3.6 support would cost dataclasses and
MAVSDK and buy nothing (3.6 is only needed for jetson-inference's prebuilt JP4 bindings).

Step-by-step setup: `docs/BRINGUP_CHECKLIST.md`.

## 9. Safety

**Non-negotiables**

1. A pilot with an RC transmitter flies every sortie, and can take over at any time.
   - Set **`COM_RC_OVERRIDE = 3`**. The PX4 default (1) does *not* let stick movement override Offboard.
   - Map a **mode switch** with Position mode on it. Flipping it ends Offboard; our app detects the mode
     change and stops sending for good (tested: `test_pilot_override_stops_streaming_and_never_reengages`).
   - Map a **kill switch** (`RC_MAP_KILL_SW`) and a return switch (`RC_MAP_RETURN_SW`); both are
     unassigned by default.
2. Failsafes configured and tested on the bench:
   - RC loss: `NAV_RCL_ACT=2` Return, `COM_RC_LOSS_T=0.5`, keep `COM_RCL_EXCEPT` bit 2 **clear** on
     the real drone (the sim sets it).
   - Offboard loss (our process dies): `COM_OF_LOSS_T=1.0`, `COM_OBL_RC_ACT=0` (Position, so the pilot
     has the sticks) or 4 Land.
   - Geofence: `GF_MAX_HOR_DIST=30`, `GF_MAX_VER_DIST=15` (both 0 = disabled by default),
     `GF_ACTION=2` Hold or 3 Return.
3. Our software's own layers (all unit-tested, `jetson/bridge.py`, `jetson/controller.py`):
   - Target lost: hover immediately. After 5 s: land, RTL or hold (`--lost-action`); latched.
   - Never found a face: hover, then the same action after 30 s.
   - Stale frames: an observation older than 0.5 s counts as "no target".
   - Perception slow: commands older than 0.5 s are replaced by hover at the 20 Hz stream.
   - Perception dead: no command for 3 s triggers land (`--stall-action`).
   - Process dead: the setpoint stream stops and PX4's offboard-loss failsafe takes over after 1 s.
   - Limits: speed 1 m/s forward, 0.5 m/s backward (blind), 30 deg/s yaw.
   - Distance: never fly forward when the face is closer than 2.5 m.
   - Vertical control off by default: never descend toward a person.
4. Default is dry-run; `--fly` prints a warning and a 5 s countdown.

**Staged test plan** (do not skip stages; log every run with `--csv` and keep the logs)

| Stage | Where | What passes |
|---|---|---|
| 0 | Laptop | `pytest jetson/tests` green; toy demo converges; SITL follow works with the sim/ track |
| 1 | Bench, **props off**, Jetson + Pixhawk on USB/battery | `linkcheck` shows mode/arm changes; `follow` dry-run commands make sense as a teammate walks by; HUD sane |
| 2 | Bench, **props off**, `--fly` | OFFBOARD engages; QGC MAVLink inspector shows `SET_POSITION_TARGET_LOCAL_NED`; RC mode switch ends it; kill switch works; pulling the camera hovers then lands; killing the app triggers the offboard-loss failsafe |
| 3 | Tethered or net cage, props on, 1-2 m | hover in OFFBOARD, slow yaw tracking only (`--max-forward 0`) |
| 4 | Open field (SBU R&D Park flying field), >= 10 m from non-participants | yaw + forward at 0.5 m/s, standoff 5 m, altitude 3-4 m, only consenting team members as targets |
| 5 | Same | full limits, walking target, lost-target -> land drill |

## 10. Legal and privacy (planning notes, not legal advice)

- **SBU's Drone Safety Policy** (Dec 2024, EH&S): buying or flying a drone on university business must
  be **pre-approved by Environmental Health & Safety**; flights over university property are prohibited
  unless approved; recreational flying only in an approved "Drone Zone". Contact on the EH&S drone page:
  file the Drone Application form. SBU's Research Security guidance also says drone software must be
  reviewed by **DoIT Information Security**, and federally funded research must avoid drones from
  designated foreign entities (does that cover Holybro? unverified; ask if any grant money is involved).
- **Part 107 vs recreational.** Federal law (49 U.S.C. 44809 and the 2018 Act section 350) lets
  institutions fly educational/research missions under recreational rules, but **SBU's policy requires
  Part 107 for educational, research or work-related flights**. So: at least one member (16+) or the
  advisor gets a **Part 107 remote pilot certificate** (knowledge test ~$175, unverified) and directly
  supervises others; everyone also takes **TRUST** (free, online, keep the certificate).
- **Registration:** the X500 is > 250 g, so register it ($5, 3 years; Part 107 registration is per
  aircraft) and mark the number on the outside.
- **Remote ID** (14 CFR 89, enforced since 16 Mar 2024): a home-built drone complies with a **broadcast
  module on the FAA's accepted list**, and the pilot must keep it in visual line of sight. Or fly inside
  an FAA-Recognized Identification Area: there is one at SBU, **"Stony Brook University Research and
  Educational Flying Field"** (2026-EDU-00046-FRIA, R&D Park, valid to 2030-03-31; the FAA data
  places it near Development Drive). Ask EH&S whether it is their "Drone Zone".
- **Airspace:** below 400 ft over campus and the R&D Park is Class G per FAA data (Long Island MacArthur's
  Class C starts at 1,500 ft MSL there; no UAS facility-map grids), so no LAANC needed. Watch for
  helicopters: the SBU Hospital heliport is ~0.8 km from main campus, ~2.6 km from the FRIA.
- **People:** Part 107.39 bars flying over anyone not directly participating (our 2 kg home-built fits no
  "operations over people" category). The followed teammate is briefed crew; keep >= 25 ft (7.6 m)
  lateral from everyone else (AMA guideline) and never over moving vehicles.
- **Privacy:**
  - We do face *detection*, not recognition. No identity templates, no face database.
  - Frames are processed in RAM.
  - `follow --out` blurs every non-target face by default. Record only consenting team members,
    delete promptly, and never film where people expect privacy (SBU policy; NY Penal Law 250.45).
  - Get written consent before any promotional video (NY Civil Rights Law 50/51).
  - If results will be published or a dataset collected, file a human-subjects determination with the
    SBU IRB (myResearch) first.

Sources: ecfr.gov / law.cornell.edu (14 CFR 48, 89, 107; 49 U.S.C. 44809 and note); FAA Section 350
Q&A (faasafety.gov, May 2022); stonybrook.edu/policy (Drone Safety Policy) and stonybrook.edu/ehs drone
page; FAA FRIA and UAS facility map data (ArcGIS); modelaircraft.org AMA Safety Code; NTIA Voluntary
Best Practices for UAS Privacy (2016). faa.gov pages were blocked to our fetcher; FAA web wording came
from search snippets (unverified). Accessed 2026-11-10.

## 11. Open questions for Eddie

1. Does the club already own an original Jetson Nano (or anything else Jetson)? Decides whether we
   buy an Orin now ($399, backordered) or start on the Nano.
2. Budget owner and purchasing route (SBU EH&S pre-approval is needed *before buying*).
3. Who becomes the Part 107 remote pilot in command?
4. Are we OK with SCRFD's research-only and YOLO-face's AGPL licences for internal work? (YuNet is MIT.)
5. Ask the sim/ track for a **face target in Gazebo** (a textured billboard or actor with a
   public-domain face photo), so `follow --source sim --fly` can be tested end to end.
