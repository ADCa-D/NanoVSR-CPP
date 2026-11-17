# 会被 echo 生成的垃圾文件：`SPIR-V`（14 字节）

## 现象
仓库根目录有个 14 字节的文件叫 `SPIR-V`，内容是一行 `[1/2] GLSL -`，没有任何代码引用它；
删了之后只要再跑一次 `build_release.bat` 又会原地复活。它是被 git 跟踪的（`git log` 显示随 release 配置那次提交进来）。

## 根因
`build_release.bat` 第 17 行写的是

    echo [1/2] GLSL -> SPIR-V

cmd 把 `>` 当成重定向：整条命令变成「把 `[1/2] GLSL -` 写进文件 `SPIR-V`」，
所以每次打包都生成这个垃圾文件，而且屏幕上永远不打印这一行。
（`build.bat` 里写的是 `echo [1/3] GLSL to SPIR-V`，用 `to`，所以一直没事——两个脚本的措辞不一致是诱因。）

## 修法
把 `build_release.bat` 里那一行改成 `echo [1/2] GLSL to SPIR-V`，删掉 `SPIR-V`，
再分别跑 `build_release.bat` 和 `build.bat` 验证文件不再出现。

## 教训
- cmd/bat 里 `echo ... -> ...`、`echo ... > ...`、`echo ... 2>&1` 都会被解析成重定向；提示文案里别出现 `>`。
- 仓库里出现「内容像一行日志、名字像产物」的文件，先怀疑脚本重定向，而不是先怀疑 git 配置。
- 现在 `grep -n "echo" *.bat | grep -- "->"` 可以当 lint 用（当前 0 命中）。