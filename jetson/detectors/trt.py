"""Native TensorRT engine for YuNet or SCRFD (Jetson only). UNVERIFIED: written before hardware.

Prefer `onnx` with device="tensorrt" (ONNX Runtime's TensorRT execution provider) when NVIDIA's
onnxruntime-gpu wheel is available for your JetPack: same speed, far less code. Use this backend
when it isn't (e.g. a bare JetPack 4.6 Nano without that wheel) or to squeeze out the last ms.

Build the engine ON THE JETSON (engines are tied to the GPU + TensorRT version):

    # Orin / JetPack 6 (TensorRT 10):
    /usr/src/tensorrt/bin/trtexec --onnx=jetson/models/face_detection_yunet_2023mar.onnx \
        --saveEngine=jetson/models/yunet_640_fp16.engine --fp16
    # SCRFD has a dynamic input: fix its shape when building
    /usr/src/tensorrt/bin/trtexec --onnx=jetson/models/scrfd_500m_bnkps.onnx \
        --saveEngine=jetson/models/scrfd_640x480_fp16.engine --fp16 \
        --shapes=input.1:1x3x480x640

Needs: tensorrt (JetPack's python3-libnvinfer) and pycuda (pip install pycuda; compiles on the
Jetson). Supports TensorRT 8.x (binding API, JetPack 4/5) and 8.5+/10.x (named-tensor API, JetPack 6).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ..face import Face
from . import decode
from .base import BackendUnavailable, FaceDetector, require_file


class TensorRTFaceDetector(FaceDetector):
    name = "tensorrt"

    def __init__(self, engine: str, family: str = "auto", score_threshold: float = 0.6,
                 nms_threshold: float = None, top_k: int = 50, input_width: int = 0) -> None:
        try:
            import tensorrt as trt
        except ImportError as e:
            raise BackendUnavailable("tensorrt Python bindings not found (Jetson only: they come with "
                                     "JetPack; inside a venv use --system-site-packages)") from e
        try:
            import pycuda.autoinit  # noqa: F401 - creates the CUDA context
            import pycuda.driver as cuda
        except ImportError as e:
            raise BackendUnavailable("pycuda not installed: pip install pycuda (on the Jetson)") from e
        self._trt, self._cuda = trt, cuda
        self.engine_path = require_file(engine, "TensorRT engine")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise BackendUnavailable(f"could not load {engine}: built for a different GPU/TensorRT version?")
        self.context = self.engine.create_execution_context()
        self.io = self._describe_io()
        inputs = [t for t in self.io if t["is_input"]]
        if len(inputs) != 1:
            raise BackendUnavailable(f"expected 1 input tensor, got {len(inputs)}")
        self.input = inputs[0]
        self.outputs = [t for t in self.io if not t["is_input"]]
        _, _, h, w = self.input["shape"]
        if h <= 0 or w <= 0:
            raise BackendUnavailable("engine has a dynamic input shape; rebuild with --shapes=...")
        self.size = (int(w), int(h))
        names = [t["name"] for t in self.outputs]
        self.family = family if family != "auto" else ("yunet" if any(n.startswith("cls_") for n in names)
                                                       else "scrfd")
        self.score_threshold, self.top_k = score_threshold, top_k
        self.nms_threshold = nms_threshold if nms_threshold is not None else (0.3 if self.family == "yunet"
                                                                               else 0.4)
        for t in self.io:  # page-locked host buffers + device buffers, allocated once
            t["host"] = cuda.pagelocked_empty(int(np.prod(t["shape"])), t["dtype"])
            t["device"] = cuda.mem_alloc(t["host"].nbytes)
        self.bindings = [int(t["device"]) for t in self.io]
        self._v3 = hasattr(self.context, "execute_async_v3")  # TensorRT >= 8.5 (required in 10.x)
        if self._v3:
            for t in self.io:
                self.context.set_tensor_address(t["name"], int(t["device"]))
            self._stream = cuda.Stream()

    def _describe_io(self) -> List[dict]:
        trt, eng = self._trt, self.engine
        io = []
        if hasattr(eng, "num_io_tensors"):  # TensorRT >= 8.5
            for i in range(eng.num_io_tensors):
                name = eng.get_tensor_name(i)
                io.append({"name": name, "is_input": eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                           "shape": tuple(eng.get_tensor_shape(name)),
                           "dtype": trt.nptype(eng.get_tensor_dtype(name))})
        else:  # TensorRT 8.0 - 8.4 (JetPack 4.6 has 8.2)
            for i in range(eng.num_bindings):
                io.append({"name": eng.get_binding_name(i), "is_input": eng.binding_is_input(i),
                           "shape": tuple(eng.get_binding_shape(i)),
                           "dtype": trt.nptype(eng.get_binding_dtype(i))})
        return io

    def _infer(self, blob: np.ndarray) -> Dict[str, np.ndarray]:
        cuda = self._cuda
        np.copyto(self.input["host"], blob.ravel().astype(self.input["dtype"], copy=False))
        cuda.memcpy_htod(self.input["device"], self.input["host"])
        if self._v3:
            ok = self.context.execute_async_v3(self._stream.handle)
            self._stream.synchronize()
        else:
            ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        out = {}
        for t in self.outputs:
            cuda.memcpy_dtoh(t["host"], t["device"])
            out[t["name"]] = t["host"].reshape(t["shape"]).astype(np.float32)
        return out

    def detect(self, frame: np.ndarray) -> List[Face]:
        iw, ih = self.size
        canvas, scale = decode.letterbox(frame, (iw, ih))
        if self.family == "yunet":
            outs = self._infer(decode.yunet_blob(canvas))
            return decode.decode_yunet(outs, iw, ih, self.score_threshold, scale, self.nms_threshold, self.top_k)
        outs = self._infer(decode.scrfd_blob(canvas))
        return decode.decode_scrfd(list(outs.values()), iw, ih, self.score_threshold, scale,
                                   self.nms_threshold, self.top_k)

    def info(self) -> Dict[str, object]:
        return {"backend": f"{self.name}:{self.family}", "model": os.path.basename(self.engine_path),
                "input": f"{self.size[0]}x{self.size[1]}", "device": "gpu",
                "tensorrt": self._trt.__version__}
