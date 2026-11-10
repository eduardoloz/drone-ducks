# Jetson bring-up checklist (the day the hardware arrives)

_Written 2026-11-10, before any hardware. Commands come from NVIDIA/PX4/MAVSDK docs and community
sources, not from running them on a Jetson. Mark each box, and fix this file where reality differs._
**Props stay off until stage 3 of the test plan** (`WHAT_WE_NEED.md` section 9).

## 0. Before it arrives (laptop)

- [ ] `uv venv --python 3.12 jetson/.venv && uv pip install --python jetson/.venv/bin/python -r jetson/requirements.txt`
- [ ] `jetson/.venv/bin/python -m jetson.fetch models && jetson/.venv/bin/python -m jetson.fetch media`
- [ ] `jetson/.venv/bin/python -m pytest jetson/tests -q`: all green
- [ ] `python -m jetson.demo --source toy --out jetson/out/toy.mp4`: watch the drone settle at ~4-5 m
- [ ] Buy: Jetson (Orin Nano Super), IMX219 camera *that says IMX219*, 15-to-22-pin cable (Orin),
      64 GB A2 microSD or NVMe, regulator (12 V for Orin / 5 V >= 5 A BEC for Nano), M3 dampers,
      JST-GH 6-pin to Dupont jumper for TELEM2. EH&S pre-approval first (WHAT_WE_NEED section 10).

## 1A. Orin Nano Super: flash and first boot (JetPack 6.2.x)

- [ ] Flash the **JetPack 6.2.1 SD card image** for "Jetson Orin Nano Developer Kit" with Balena Etcher.
      If the board shows a black screen or an old UEFI, follow NVIDIA's initial-setup firmware
      update first (jetson-ai-lab.com initial setup guide).
- [ ] First boot: finish setup (user, Wi-Fi), then upgrade to the newest 6.2.x:
      `sudo apt update && sudo apt dist-upgrade && sudo reboot`
- [ ] `cat /etc/nv_tegra_release` shows R36 (6.2.3 = L4T 36.5.2). **Don't** move to JetPack 7.2 yet
      (no SD image; open UART-TX bug report on the 40-pin header).
- [ ] Power mode **MAXN SUPER**: top-bar power menu, or `cat /etc/nvpmodel.conf` to find its id, then
      `sudo nvpmodel -m <id>`. For benchmarks also `sudo jetson_clocks`. Watch temps: `tegrastats`.
- [ ] Python env (system OpenCV + TensorRT stay visible):
      ```bash
      sudo apt install -y python3-venv python3-pip git
      git clone https://github.com/eduardoloz/drone-ducks.git ~/drone-ducks && cd ~/drone-ducks
      git checkout eduardo
      python3 -m venv --system-site-packages ~/quack-venv && . ~/quack-venv/bin/activate
      pip install -r jetson/requirements-jetson-jp6.txt
      python -c "import cv2; print(cv2.__version__, cv2.__file__)"   # expect 4.8.x under /usr/lib
      ```

## 1B. Original Nano: flash and first boot (JetPack 4.6.x)

- [ ] Flash the newest **Jetson Nano Developer Kit SD card image** NVIDIA still offers (JetPack 4.6.x),
      then `sudo apt update && sudo apt upgrade`. `cat /etc/nv_tegra_release` shows R32 REVISION 7.x.
- [ ] Power from a **5 V >= 4 A barrel supply with the J48 jumper fitted** (not micro-USB).
      `sudo nvpmodel -m 0` (MAXN 10 W), `sudo jetson_clocks`; fit the 5 V fan.
- [ ] Python 3.8 env (system Python 3.6 is too old):
      ```bash
      sudo apt install -y python3.8 python3.8-venv python3.8-dev git
      git clone https://github.com/eduardoloz/drone-ducks.git ~/drone-ducks && cd ~/drone-ducks && git checkout eduardo
      python3.8 -m venv ~/quack-venv && . ~/quack-venv/bin/activate && pip install -U pip
      pip install -r jetson/requirements-jetson-jp4.txt
      ```
- [ ] Optional GPU route: Ultralytics JetPack 4 container (Python 3.8, TensorRT 8.2, onnxruntime-gpu 1.8):
      ```bash
      t=ultralytics/ultralytics:latest-jetson-jetpack4
      sudo docker pull $t
      sudo docker run -it --ipc=host --runtime=nvidia -v ~/drone-ducks:/drone-ducks $t
      # inside: pip install mavsdk==4.0.6 && cd /drone-ducks && python3 -m jetson.doctor
      ```

## 2. Package sanity (both)

- [ ] `python -m jetson.fetch models && python -m jetson.fetch media` (or copy `jetson/models/` over)
- [ ] `python -m jetson.doctor`: no FAIL lines; note which backends say OK
- [ ] `python -m pytest jetson/tests -q`: all green on the device

## 3. Camera

- [ ] Power off, connect the CSI ribbon (contacts toward the connector's contacts; Orin needs the
      22-pin end on the board side). Power on.
- [ ] Orin: if `/dev/video0` is missing, run `sudo /opt/nvidia/jetson-io/jetson-io.py`, configure the
      CSI connector for "Camera IMX219", reboot (per NVIDIA docs; verify).
- [ ] Capture test, headless:
      `gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=60 ! 'video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1' ! fakesink`.
      Note the sensor-mode list it prints, and pick a **full-FOV** mode (1640x1232 on IMX219).
- [ ] Orin (OpenCV has GStreamer): `python -m jetson.demo --source csi://0 --hfov 62.2 --max-frames 300 --out jetson/out/csi.mp4`
- [ ] Nano (pip OpenCV has no GStreamer): start the hardware-JPEG MJPEG server in a tmux pane, then
      read it:
      ```bash
      tmux new-session -d -s cam "$(python -c 'from jetson.sources import csi_mjpeg_server_cmd as c; print(c())')"
      python -m jetson.demo --source tcp://127.0.0.1:5000 --hfov 62.2 --max-frames 300 --out jetson/out/csi.mp4
      ```
      (USB webcam alternative: `--source /dev/video0`.)
- [ ] **Calibrate distance:** a teammate stands at 3.0 m and 5.0 m; read `dist` on the HUD/CSV.
      Adjust `--face-width` until both read right. Record the value in `WHAT_WE_NEED.md`.

## 4. Benchmarks on the real device

- [ ] `sudo jetson_clocks`, then `python -m jetson.bench --save`. Commit `jetson/bench_results/<jetson>_<date>.md`
      and write the power mode into it.
- [ ] Orin GPU via ONNX Runtime:
      ```bash
      pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
      python -c "import onnxruntime as o; print(o.get_available_providers())"   # want TensorrtExecutionProvider
      python -m jetson.bench --backends onnx:yunet@tensorrt,onnx:yunet@cuda,yunet --widths 640 --save
      ```
      (The first TensorRT run builds an engine into `jetson/models/trt_cache/` and takes minutes.)
- [ ] Native TensorRT (optional):
      `/usr/src/tensorrt/bin/trtexec --onnx=jetson/models/face_detection_yunet_2023mar.onnx --saveEngine=jetson/models/yunet_640_fp16.engine --fp16`,
      `pip install pycuda`, `python -m jetson.bench --backends tensorrt --widths 640`
- [ ] jetson-inference (optional): `git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference`,
      `cd jetson-inference && docker/run.sh` (JetPack 6.2: tag `r36.3.0` may need passing by hand),
      mount `~/drone-ducks`, then `python3 -m jetson.bench --backends jetson_inference`.
- [ ] Glass-to-glass latency: film a phone stopwatch with the camera and screen-record the HUD; the
      difference is camera + processing latency. Put the number in `ARCHITECTURE.md` section 4.

## 5. Pixhawk link (props OFF, drone on the bench)

- [ ] Wire TELEM2 -> header: 6C pin 2 TX -> Jetson pin 10, 6C pin 3 RX -> Jetson pin 8, GND -> pin 6.
      Leave 6C pin 1 (+5 V) unconnected.
- [ ] Nano only: `sudo systemctl disable --now nvgetty`. Both: `sudo usermod -aG dialout $USER`, log out/in.
- [ ] In QGroundControl (USB or radio), set and reboot the Pixhawk:
      `MAV_1_CONFIG=102`, `MAV_1_MODE=2`, `SER_TEL2_BAUD=921600`, `COM_RC_OVERRIDE=3`,
      `GF_MAX_HOR_DIST=30`, `GF_MAX_VER_DIST=15`, `GF_ACTION=2`, kill switch + return switch assigned,
      `COM_RCL_EXCEPT` bit 2 clear, `COM_OBL_RC_ACT=0` (or 4).
- [ ] `python -m jetson.linkcheck --connect serial:///dev/ttyTHS1:921600`: heartbeat. Flip the RC mode
      switch (mode changes), arm/disarm props-off (armed changes), lift the drone (rel_alt changes).
- [ ] If QGC over Wi-Fi is wanted too, install mavlink-router (config in `WHAT_WE_NEED.md` section 7)
      and use `--connect udpin://0.0.0.0:14540` instead.

## 6. Dry run with everything live (no commands sent)

- [ ] `python -m jetson.follow --source csi://0 --csv jetson/out/dry.csv --out jetson/out/dry.mp4`
      (Nano: `--source tcp://127.0.0.1:5000`). A consenting teammate walks left/right/closer/away.
      Check in the log: yaw sign follows them, forward goes negative when they are closer than 4 m,
      LOST -> TIMEOUT when they leave the frame. Bystander faces in `dry.mp4` are blurred.

## 7. Props-off OFFBOARD test (stage 2)

- [ ] Props **off**, battery in, RC on, pilot ready. Then:
      `python -m jetson.follow --source csi://0 --fly --bench-test --connect serial:///dev/ttyTHS1:921600 --max-forward 0 --csv jetson/out/bench.csv`
- [ ] Pilot arms. The app enters OFFBOARD. QGC MAVLink Inspector shows `SET_POSITION_TARGET_LOCAL_NED`
      at ~20 Hz.
- [ ] Each of these must work and be logged:
  - [ ] RC mode switch to Position: the app prints "left OFFBOARD", stops for good.
  - [ ] Kill switch stops the motors.
  - [ ] Cover the camera: LOST, then land after 5 s.
  - [ ] Unplug the camera: hover, then land after 3 s.
  - [ ] `kill -9` the app: PX4 offboard-loss failsafe after 1 s.

## 8. Before the first flight

- [ ] EH&S approval in hand, Part 107 RPIC present, Remote ID module registered, TRUST certificates,
      site = SBU R&D Park flying field, visual observer, only consenting team members as targets.
- [ ] Stage 3 (tethered/cage, `--max-forward 0`), then stages 4-5 in the open field.
- [ ] After every sortie: keep the `--csv` log and a note of what happened; delete any recording that
      isn't needed.
