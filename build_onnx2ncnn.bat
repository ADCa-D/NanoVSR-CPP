@echo off
rem Build onnx2ncnn from source (needs cmake + protobuf).
rem Steps: cmake -> ncnn source -> vcpkg(protobuf) -> cmake build target onnx2ncnn
rem Everything is logged to build_onnx2ncnn.log
setlocal
set ROOT=D:\nanovsr-cpp\_trtbuild
set LOG=%ROOT%\build.log
if not exist "%ROOT%" mkdir "%ROOT%"
cd /d "%ROOT%"

echo [1/6] cmake >> "%LOG%"
where cmake >nul 2>nul || winget install --id Kitware.CMake --accept-source-agreements --accept-package-agreements --disable-interactivity >> "%LOG%" 2>&1
set PATH=C:\Program Files\CMake\bin;%PATH%

echo [2/6] ncnn source >> "%LOG%"
if not exist ncnn-src (
  git clone --depth 1 https://ghfast.top/https://github.com/Tencent/ncnn.git ncnn-src >> "%LOG%" 2>&1
)

echo [3/6] vcpkg >> "%LOG%"
if not exist vcpkg (
  git clone --depth 1 https://ghfast.top/https://github.com/microsoft/vcpkg.git vcpkg >> "%LOG%" 2>&1
  call vcpkg\bootstrap-vcpkg.bat >> "%LOG%" 2>&1
)

echo [4/6] protobuf >> "%LOG%"
if not exist vcpkg_installed (
  vcpkg\vcpkg.exe install protobuf:x64-windows >> "%LOG%" 2>&1
)

echo [5/6] configure >> "%LOG%"
cmake -S ncnn-src -B build -G "Visual Studio 17 2022" -A x64 ^
  -DNCNN_BUILD_TOOLS=ON -DNCNN_VULKAN=OFF -DNCNN_BUILD_BENCHMARK=OFF ^
  -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF ^
  -DNCNN_SYSTEM_PROTOBUF=ON ^
  -DCMAKE_TOOLCHAIN_FILE=%ROOT%\vcpkg\scripts\buildsystems\vcpkg.cmake >> "%LOG%" 2>&1

echo [6/6] build onnx2ncnn >> "%LOG%"
cmake --build build --config Release --target onnx2ncnn >> "%LOG%" 2>&1

if exist build\tools\onnx\Release\onnx2ncnn.exe (
  echo OK: build\tools\onnx\Release\onnx2ncnn.exe >> "%LOG%"
) else (
  echo FAILED: see %LOG% >> "%LOG%"
)
