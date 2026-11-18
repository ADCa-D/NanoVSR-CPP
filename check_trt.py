import importlib


for m in ("tensorrt", "tensorrt_libs", "onnx", "onnxruntime", "torch"):
    try:
        mod = importlib.import_module(m)
        print("%-16s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("%-16s FAIL %s: %s" % (m, type(e).__name__, str(e)[:70]))
