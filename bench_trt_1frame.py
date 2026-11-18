# -*- coding: utf-8 -*-
"""单帧链的 ONNX 推理测速：CUDA EP vs TensorRT EP，尺寸取 272x480（真实工作点）。"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\nanovsr-cpp")
sys.path.insert(0, r"D:\nanovsr")
from export_weights import build  # noqa: E402

H, W = 272, 480
ONNX = r"D:\nanovsr-cpp\bench_1frame_272x480.onnx"


class Frame(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, feat, prop_f, prop_b, x4):
        m = self.m
        fwd = m.forward_net(feat + prop_f)
        bwd = m.backward_net(feat + prop_b)
        fused = m.fusion(torch.cat([fwd, bwd], 1))
        u = m.upsample2(m.upsample1(fused))
        base = F.interpolate(x4, scale_factor=4, mode="bilinear", align_corners=False)
        return (m.conv_last(u) + base).clamp(0, 1), fwd, bwd


def main():
    m, nf, nb = build("1.7m")
    net = Frame(m).cuda().eval()
    feat = torch.randn(1, nf, H, W, device="cuda")
    pf = torch.zeros(1, nf, H, W, device="cuda")
    pb = torch.zeros(1, nf, H, W, device="cuda")
    x4 = torch.rand(1, 3, H, W, device="cuda")

    if not os.path.exists(ONNX):
        torch.onnx.export(net, (feat, pf, pb, x4), ONNX,
                          input_names=["feat", "prop_f", "prop_b", "x4"],
                          output_names=["y", "fwd", "bwd"], opset_version=17, dynamo=False)
    print("onnx:", os.path.getsize(ONNX) // 1024, "KB")

    try:
        import tensorrt_libs
        os.environ["PATH"] = os.path.dirname(tensorrt_libs.__file__) + os.pathsep + os.environ["PATH"]
    except Exception:
        pass

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    feeds = {"feat": feat.cpu().numpy(), "prop_f": pf.cpu().numpy(),
             "prop_b": pb.cpu().numpy(), "x4": x4.cpu().numpy()}

    with torch.inference_mode():
        for _ in range(3):
            net(feat, pf, pb, x4)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            net(feat, pf, pb, x4)
        torch.cuda.synchronize()
        print("torch fp32    : %.1f ms/帧" % ((time.time() - t0) / 10 * 1000))

    for prov in (["CUDAExecutionProvider"], ["TensorrtExecutionProvider", "CUDAExecutionProvider"]):
        tag = prov[0]
        try:
            t0 = time.time()
            sess = ort.InferenceSession(ONNX, so, providers=prov)
            print("%-14s session %.1f s" % (tag, time.time() - t0))
            for _ in range(3):
                sess.run(None, feeds)
            t0 = time.time()
            for _ in range(10):
                sess.run(None, feeds)
            ms = (time.time() - t0) / 10 * 1000
            print("%-14s %.1f ms/帧   实际 %s" % (tag, ms, sess.get_providers()[0]))
        except Exception as e:
            print("%-14s FAIL %s: %s" % (tag, type(e).__name__, str(e)[:120]))


if __name__ == "__main__":
    main()
