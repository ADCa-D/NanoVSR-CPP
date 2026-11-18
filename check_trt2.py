import os
import sys

import onnxruntime as ort

print("ORT", ort.__version__)
print("providers:", ort.get_available_providers())

# tensorrt_libs 里的 DLL 是带版本号的（nvinfer_10.dll），ORT 期望无版本号的名字
try:
    import tensorrt_libs

    d = os.path.dirname(tensorrt_libs.__file__)
    print("trt libs:", d)
    print("on PATH:", any(p.lower() == d.lower() for p in os.environ.get("PATH", "").split(os.pathsep)))
except Exception as e:
    print("tensorrt_libs fail:", e)
