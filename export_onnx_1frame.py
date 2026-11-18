# -*- coding: utf-8 -*-
"""把 NanoVSR 的「单帧链」导出成 ONNX，并用 ORT 的 TensorRT EP 跑一遍，
和 torch 输出对比。目的：判断 TRT 迁移是否值得。

单帧链（与 engine 里的时序递推对应）：
    in   = feat_t + prop_f
    fwd  = forward_net(in)            # 20 blocks
    bwd  = backward_net(feat_t + prop_b)
    fused= fusion(cat(fwd,bwd))
    y    = conv_last(upsample2(upsample1(fused))) + bilinear4x(x4)
"""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\nanovsr-cpp")
sys.path.insert(0, r"D:\nanovsr")
from export_weights import build  # noqa: E402

H = W = 64
ONNX = r"D:\nanovsr-cpp\nanovsr_1frame.onnx"


class Frame(nn.Module):
    """单帧链：输入 feat/prop_f/prop_b/x4，输出 y/fwd/bwd。"""

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
    m = m.cuda().eval()
    net = Frame(m).cuda().eval()

    feat = torch.randn(1, nf, H, W, device="cuda")
    pf = torch.randn(1, nf, H, W, device="cuda")
    pb = torch.randn(1, nf, H, W, device="cuda")
    x4 = torch.rand(1, 3, H, W, device="cuda")

    with torch.inference_mode():
        y, fwd, bwd = net(feat, pf, pb, x4)

    print("exporting onnx ...")
    torch.onnx.export(net, (feat, pf, pb, x4), ONNX,
                      input_names=["feat", "prop_f", "prop_b", "x4"],
                      output_names=["y", "fwd", "bwd"],
                      opset_version=17, dynamo=False)
    print("onnx:", os.path.getsize(ONNX) // 1024, "KB")

    # TRT 的 DLL 目录
    try:
        import tensorrt_libs
        os.environ["PATH"] = os.path.dirname(tensorrt_libs.__file__) + os.pathsep + os.environ["PATH"]
    except Exception as e:
        print("tensorrt_libs:", e)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    feeds = {"feat": feat.cpu().numpy(), "prop_f": pf.cpu().numpy(),
             "prop_b": pb.cpu().numpy(), "x4": x4.cpu().numpy()}

    for prov in (["CUDAExecutionProvider"],
                 ["TensorrtExecutionProvider", "CUDAExecutionProvider"]):
        try:
            sess = ort.InferenceSession(ONNX, so, providers=prov)
            out = sess.run(None, feeds)
            dy = float(np.abs(out[0] - y.cpu().numpy()).max())
            df = float(np.abs(out[1] - fwd.cpu().numpy()).max())
            print("%-42s ok  used=%s" % (str(prov), sess.get_providers()[0]))
            print("    maxdiff  y=%.3e  fwd=%.3e" % (dy, df))
        except Exception as e:
            print("%-42s FAIL %s: %s" % (str(prov), type(e).__name__, str(e)[:160]))


if __name__ == "__main__":
    main()
