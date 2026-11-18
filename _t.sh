export MSYS2_ARG_CONV_EXCL='*'
cmd.exe /c "nanovsr.exe --in kn_in.mp4 --model-size 644k --limit 1 --crf 0 --codec h264 --tile 200x144 --out o0.mp4" > o0.log 2>&1; echo e0=$?
cmd.exe /c "nanovsr.exe --in kn_in.mp4 --model-size 644k --limit 1 --crf 0 --codec h264 --tile 200x144 --ov 64 --out o64.mp4" > o64.log 2>&1; echo e64=$?
grep -a 'tile \|总处理量' o0.log o64.log | tr -d '\r'
D:/ffmpeg-9.0.1/ffmpeg.exe -y -v error -i o0.mp4 -vframes 1 f0.png
D:/ffmpeg-9.0.1/ffmpeg.exe -y -v error -i o64.mp4 -vframes 1 f64.png
python psnr.py f0.png f64.png
python -c "from PIL import Image; import numpy as np; a=np.asarray(Image.open('f0.png').convert('RGB'),dtype=np.float32); b=np.asarray(Image.open('f64.png').convert('RGB'),dtype=np.float32); d=np.abs(a-b).mean(axis=2); col=d.mean(axis=0); import sys; idx=np.argsort(col)[-6:]; print('worst cols', sorted(int(i) for i in idx), 'max col mean', round(float(col.max()),2)); row=d.mean(axis=1); r=np.argsort(row)[-6:]; print('worst rows', sorted(int(i) for i in r), 'max row mean', round(float(row.max()),2)); print('global mean diff', round(float(d.mean()),3))"