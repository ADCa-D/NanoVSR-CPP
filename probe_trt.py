# -*- coding: utf-8 -*-
"""TRT 可行性探针：把 tensorrt_libs 加进 PATH，用 ORT 的 TRT EP 跑一个最小卷积。"""
import os
import sys

import numpy as np
import torch

# 1) DLL 目录进 PATH（nvinfer_10.dll 等都在这里）
try:
    import tensorrt_libs

    d = os.path.dirname(tensorrt_libs.__file__)
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    print("PATH += ", d)
except Exception as e:
    print("tensorrt_libs 不可用:", e)

import onnxruntime as ort  # noqa: E402

print("ORT", ort.__version__)
print("providers:", ort.get_available_providers())

# 2) 造一个最小 ONNX（3x3 卷积，64->64，与真实层同形状）
m = torch.nn.Conv2d(64, 64, 3, padding=1).cuda().eval()
x = torch.randn(1, 64, 64, 64, device="cuda")
torch.onnx.export(m, x, "probe_conv.onnx", input_names=["x"], output_names=["y"],
                  opset_version=17, dynamo=False)
print("onnx exported")

# 3) 先用 CUDA EP 跑一遍做基准
so = ort.SessionOptions()
so.log_severity_level = 3
for prov in (["CUDAExecutionProvider"], ["TensorrtExecutionProvider", "CUDAExecutionProvider"]):
    try:
        sess = ort.InferenceSession("probe_conv.onnx", so, providers=prov)
        xn = x.cpu().numpy()
        y = sess.run(None, {"x": xn})[0]
        used = sess.get_providers()
        print("%-46s -> ok, 实际使用 %s, 输出均值 %.5f" % (str(prov), used, float(y.mean())))
    except Exception as e:
        print("%-46s -> FAIL %s: %s" % (str(prov), type(e).__name__, str(e)[:120]))
