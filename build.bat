@echo off
rem Build: GLSL -> SPIR-V (glslang), then C++ hosts (cl).
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
set HERE=%~dp0
if "%VKROOT%"=="" set VKROOT=D:\nanovsr-webui\vk
set INC=%VKROOT%\sdk\Vulkan-Headers-main\include
set GLSLANG=%VKROOT%\glslang\bin\glslang.exe

if not exist "%GLSLANG%" (
  echo [ERROR] glslang not found: %GLSLANG%
  exit /b 1
)

echo [1/2] GLSL to SPIR-V
"%GLSLANG%" -V shaders\conv3x3_c64.comp            -o shaders\conv3x3_c64.spv            || goto :fail
"%GLSLANG%" -V shaders\conv3x3_smallc.comp         -o shaders\conv3x3_smallc.spv         || goto :fail
"%GLSLANG%" -V shaders\conv_gen.comp               -o shaders\conv_gen.spv               || goto :fail
"%GLSLANG%" -V shaders\shuffle_prelu.comp          -o shaders\shuffle_prelu.spv          || goto :fail
"%GLSLANG%" -V -DIN_F16 shaders\shuffle_prelu.comp -o shaders\shuffle_prelu_f16.spv      || goto :fail
"%GLSLANG%" -V shaders\bilinear4_add_u8.comp       -o shaders\bilinear4_add_u8.spv       || goto :fail
"%GLSLANG%" -V shaders\add.comp                    -o shaders\add.spv                    || goto :fail
"%GLSLANG%" -V shaders\cat2.comp                   -o shaders\cat2.spv                   || goto :fail
"%GLSLANG%" -V shaders\u8rgb_to_f32.comp           -o shaders\u8rgb_to_f32.spv           || goto :fail
"%GLSLANG%" -V shaders\copy.comp                   -o shaders\copy.spv                   || goto :fail

rem Tensor-core conv: one variant per model width (num_feat) and output type
"%GLSLANG%" -V --target-env vulkan1.1 shaders\conv3x3_cm.comp                      -o shaders\conv3x3_cm.spv          || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=48 shaders\conv3x3_cm.comp      -o shaders\conv3x3_cm_48.spv       || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=64 shaders\conv3x3_cm.comp      -o shaders\conv3x3_cm_64.spv       || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=96 shaders\conv3x3_cm.comp      -o shaders\conv3x3_cm_96.spv       || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=48 -DOUT_F16 shaders\conv3x3_cm.comp -o shaders\conv3x3_cm_48_f16.spv || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=64 -DOUT_F16 shaders\conv3x3_cm.comp -o shaders\conv3x3_cm_64_f16.spv || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=96 -DOUT_F16 shaders\conv3x3_cm.comp -o shaders\conv3x3_cm_96_f16.spv || goto :fail
rem Tensor-core 1x1 conv (fusion: C = 2*num_feat)
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=96  shaders\conv1x1_cm.comp -o shaders\conv1x1_cm_96.spv  || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=128 shaders\conv1x1_cm.comp -o shaders\conv1x1_cm_128.spv || goto :fail
"%GLSLANG%" -V --target-env vulkan1.1 -DC_OVERRIDE=192 shaders\conv1x1_cm.comp -o shaders\conv1x1_cm_192.spv || goto :fail

echo [2/2] C++ (model test / bench / main)
del /q nanovsr.exe 2>nul
cl /nologo /std:c++20 /utf-8 /EHsc /MD /O2 /I"%INC%" ^
   src\model_test.cpp /Fe:model_test.exe ^
   /link /LIBPATH:"%VKROOT%\vk" /LIBPATH:"%VKROOT%" vulkan-1.lib || goto :fail
cl /nologo /std:c++20 /utf-8 /EHsc /MD /O2 /I"%INC%" ^
   src\bench_model.cpp /Fe:bench_model.exe ^
   /link /LIBPATH:"%VKROOT%\vk" /LIBPATH:"%VKROOT%" vulkan-1.lib || goto :fail
cl /nologo /std:c++20 /utf-8 /EHsc /MD /O2 /I"%INC%" ^
   src\nanovsr.cpp /Fe:nanovsr.exe ^
   /link /LIBPATH:"%VKROOT%\vk" /LIBPATH:"%VKROOT%" vulkan-1.lib || goto :fail
exit /b 0

:fail
exit /b 1
