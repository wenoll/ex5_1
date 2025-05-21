# 创建 Python 3.9 虚拟环境


```python
# 安装 python3.9 及必要工具
!sudo apt-get update -y
!sudo apt-get install python3.9 python3.9-venv python3.9-distutils curl -y
# 创建虚拟环境（不会自带 pip）
!python3.9 -m venv /content/tflite_env
# 下载官方 get-pip 脚本
!curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# 使用虚拟环境中的 python 执行脚本安装 pip
!/content/tflite_env/bin/python get-pip.py
#验证 pip 是否生效
!/content/tflite_env/bin/pip --version


```

    Get:1 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ InRelease [3,632 B]
    Hit:2 http://archive.ubuntu.com/ubuntu jammy InRelease
    Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]
    Get:4 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease [1,581 B]
    Get:5 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
    Get:6 https://r2u.stat.illinois.edu/ubuntu jammy InRelease [6,555 B]
    Get:7 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]
    Hit:8 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
    Hit:9 https://ppa.launchpadcontent.net/graphics-drivers/ppa/ubuntu jammy InRelease
    Get:10 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease [24.6 kB]
    Get:11 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages [1,683 kB]
    Get:12 https://r2u.stat.illinois.edu/ubuntu jammy/main amd64 Packages [2,729 kB]
    Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [4,540 kB]
    Get:14 https://r2u.stat.illinois.edu/ubuntu jammy/main all Packages [8,958 kB]
    Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1,552 kB]
    Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [3,255 kB]
    Get:17 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [35.2 kB]
    Get:18 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [83.2 kB]
    Get:19 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [2,944 kB]
    Get:20 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [4,387 kB]
    Get:21 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1,245 kB]
    Get:22 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy/main amd64 Packages [77.3 kB]
    Fetched 31.9 MB in 8s (3,988 kB/s)
    Reading package lists... Done
    W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    curl is already the newest version (7.81.0-1ubuntu1.20).
    The following additional packages will be installed:
      libpython3.9-minimal libpython3.9-stdlib python3.9-lib2to3 python3.9-minimal
    Suggested packages:
      binfmt-support
    The following NEW packages will be installed:
      libpython3.9-minimal libpython3.9-stdlib python3.9 python3.9-distutils
      python3.9-lib2to3 python3.9-minimal python3.9-venv
    0 upgraded, 7 newly installed, 0 to remove and 91 not upgraded.
    Need to get 7,815 kB of archives.
    After this operation, 23.1 MB of additional disk space will be used.
    Get:1 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 libpython3.9-minimal amd64 3.9.22-1+jammy1 [837 kB]
    Get:2 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.9-minimal amd64 3.9.22-1+jammy1 [2,072 kB]
    Get:3 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 libpython3.9-stdlib amd64 3.9.22-1+jammy1 [1,844 kB]
    Get:4 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.9 amd64 3.9.22-1+jammy1 [93.0 kB]
    Get:5 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.9-lib2to3 all 3.9.22-1+jammy1 [127 kB]
    Get:6 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.9-distutils all 3.9.22-1+jammy1 [193 kB]
    Get:7 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.9-venv amd64 3.9.22-1+jammy1 [2,650 kB]
    Fetched 7,815 kB in 2s (3,303 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 7.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package libpython3.9-minimal:amd64.
    (Reading database ... 126102 files and directories currently installed.)
    Preparing to unpack .../0-libpython3.9-minimal_3.9.22-1+jammy1_amd64.deb ...
    Unpacking libpython3.9-minimal:amd64 (3.9.22-1+jammy1) ...
    Selecting previously unselected package python3.9-minimal.
    Preparing to unpack .../1-python3.9-minimal_3.9.22-1+jammy1_amd64.deb ...
    Unpacking python3.9-minimal (3.9.22-1+jammy1) ...
    Selecting previously unselected package libpython3.9-stdlib:amd64.
    Preparing to unpack .../2-libpython3.9-stdlib_3.9.22-1+jammy1_amd64.deb ...
    Unpacking libpython3.9-stdlib:amd64 (3.9.22-1+jammy1) ...
    Selecting previously unselected package python3.9.
    Preparing to unpack .../3-python3.9_3.9.22-1+jammy1_amd64.deb ...
    Unpacking python3.9 (3.9.22-1+jammy1) ...
    Selecting previously unselected package python3.9-lib2to3.
    Preparing to unpack .../4-python3.9-lib2to3_3.9.22-1+jammy1_all.deb ...
    Unpacking python3.9-lib2to3 (3.9.22-1+jammy1) ...
    Selecting previously unselected package python3.9-distutils.
    Preparing to unpack .../5-python3.9-distutils_3.9.22-1+jammy1_all.deb ...
    Unpacking python3.9-distutils (3.9.22-1+jammy1) ...
    Selecting previously unselected package python3.9-venv.
    Preparing to unpack .../6-python3.9-venv_3.9.22-1+jammy1_amd64.deb ...
    Unpacking python3.9-venv (3.9.22-1+jammy1) ...
    Setting up libpython3.9-minimal:amd64 (3.9.22-1+jammy1) ...
    Setting up python3.9-lib2to3 (3.9.22-1+jammy1) ...
    Setting up python3.9-distutils (3.9.22-1+jammy1) ...
    Setting up python3.9-minimal (3.9.22-1+jammy1) ...
    Setting up libpython3.9-stdlib:amd64 (3.9.22-1+jammy1) ...
    Setting up python3.9 (3.9.22-1+jammy1) ...
    Setting up python3.9-venv (3.9.22-1+jammy1) ...
    Processing triggers for man-db (2.10.2-1) ...
    Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
    Collecting pip
      Downloading pip-25.1.1-py3-none-any.whl.metadata (3.6 kB)
    Collecting wheel
      Downloading wheel-0.45.1-py3-none-any.whl.metadata (2.3 kB)
    Downloading pip-25.1.1-py3-none-any.whl (1.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m10.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading wheel-0.45.1-py3-none-any.whl (72 kB)
    Installing collected packages: wheel, pip
    [2K  Attempting uninstall: pip
    [2K    Found existing installation: pip 23.0.1
    [2K    Uninstalling pip-23.0.1:
    [2K      Successfully uninstalled pip-23.0.1
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2/2[0m [pip]
    [1A[2KSuccessfully installed pip-25.1.1 wheel-0.45.1
    pip 25.1.1 from /content/tflite_env/lib/python3.9/site-packages/pip (python 3.9)


# 安装核心依赖


```python
! /content/tflite_env/bin/pip install -q \
  tensorflow==2.10.0 \
  keras==2.10.0 \
  numpy==1.23.5 \
  protobuf==3.19.6 \
  tensorflow-hub==0.12.0 \
  tflite-support==0.4.2 \
  tensorflow-datasets==4.8.3 \
  sentencepiece==0.1.99 \
  sounddevice==0.4.5 \
  librosa==0.8.1 \
  flatbuffers==23.5.26 \
  matplotlib==3.5.3 \
  opencv-python==4.8.0.76


```

      Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m578.1/578.1 MB[0m [31m22.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m64.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m44.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.1/17.1 MB[0m [31m150.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.1/60.1 MB[0m [31m9.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.4/5.4 MB[0m [31m121.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m46.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.2/11.2 MB[0m [31m150.2 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m61.7/61.7 MB[0m [31m53.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.9/5.9 MB[0m [31m129.8 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.9/5.9 MB[0m [31m130.2 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m128.8 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.7/4.7 MB[0m [31m127.2 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.6/4.6 MB[0m [31m123.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m69.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.5/24.5 MB[0m [31m161.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.7/3.7 MB[0m [31m111.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.9/43.9 MB[0m [31m57.5 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.6/4.6 MB[0m [31m116.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m107.1 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.5/13.5 MB[0m [31m158.9 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m38.6/38.6 MB[0m [31m56.7 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m49.8 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m32.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.1/5.1 MB[0m [31m121.9 MB/s[0m eta [36m0:00:00[0m
    [?25h[33m  DEPRECATION: Building 'promise' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'promise'. Discussion can be found at https://github.com/pypa/pip/issues/6334[0m[33m
    [0m  Building wheel for promise (setup.py) ... [?25l[?25hdone
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m82/82[0m [tensorflow]
    [1A[2K

# 安装 tflite-model-maker 本体


```python
! /content/tflite_env/bin/pip install tflite-model-maker==0.4.2

```

    Collecting tflite-model-maker==0.4.2
      Downloading tflite_model_maker-0.4.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting tf-models-official==2.3.0 (from tflite-model-maker==0.4.2)
      Downloading tf_models_official-2.3.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Requirement already satisfied: numpy>=1.17.3 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (1.23.5)
    Requirement already satisfied: pillow>=7.0.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (11.2.1)
    Requirement already satisfied: sentencepiece>=0.1.91 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (0.1.99)
    Requirement already satisfied: tensorflow-datasets>=2.1.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (4.8.3)
    Collecting fire>=0.3.1 (from tflite-model-maker==0.4.2)
      Downloading fire-0.7.0.tar.gz (87 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: flatbuffers>=2.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (23.5.26)
    Requirement already satisfied: absl-py>=0.10.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (1.4.0)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 (from tflite-model-maker==0.4.2)
      Downloading urllib3-1.25.11-py2.py3-none-any.whl.metadata (41 kB)
    Requirement already satisfied: tflite-support>=0.4.2 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (0.4.2)
    Collecting tensorflowjs<3.19.0,>=2.4.0 (from tflite-model-maker==0.4.2)
      Downloading tensorflowjs-3.18.0-py3-none-any.whl.metadata (1.6 kB)
    Requirement already satisfied: tensorflow>=2.6.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (2.10.0)
    Collecting numba==0.53 (from tflite-model-maker==0.4.2)
      Downloading numba-0.53.0-cp39-cp39-manylinux2014_x86_64.whl.metadata (3.4 kB)
    Requirement already satisfied: librosa==0.8.1 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (0.8.1)
    Collecting lxml>=4.6.1 (from tflite-model-maker==0.4.2)
      Downloading lxml-5.4.0-cp39-cp39-manylinux_2_28_x86_64.whl.metadata (3.5 kB)
    Collecting PyYAML>=5.1 (from tflite-model-maker==0.4.2)
      Downloading PyYAML-6.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
    Collecting matplotlib<3.5.0,>=3.0.3 (from tflite-model-maker==0.4.2)
      Downloading matplotlib-3.4.3-cp39-cp39-manylinux1_x86_64.whl.metadata (5.7 kB)
    Requirement already satisfied: six>=1.12.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (1.17.0)
    Collecting tensorflow-addons>=0.11.2 (from tflite-model-maker==0.4.2)
      Downloading tensorflow_addons-0.23.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.8 kB)
    Collecting neural-structured-learning>=1.3.1 (from tflite-model-maker==0.4.2)
      Downloading neural_structured_learning-1.4.0-py2.py3-none-any.whl.metadata (2.5 kB)
    Collecting tensorflow-model-optimization>=0.5 (from tflite-model-maker==0.4.2)
      Downloading tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
    Collecting Cython>=0.29.13 (from tflite-model-maker==0.4.2)
      Downloading cython-3.1.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.5 kB)
    Collecting scann==1.2.6 (from tflite-model-maker==0.4.2)
      Downloading scann-1.2.6-cp39-cp39-manylinux2014_x86_64.whl.metadata (4.5 kB)
    Requirement already satisfied: tensorflow-hub<0.13,>=0.7.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-model-maker==0.4.2) (0.12.0)
    Requirement already satisfied: audioread>=2.0.0 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (3.0.1)
    Requirement already satisfied: scipy>=1.0.0 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (1.13.1)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (1.6.1)
    Requirement already satisfied: joblib>=0.14 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (1.5.0)
    Requirement already satisfied: decorator>=3.0.0 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (5.2.1)
    Requirement already satisfied: resampy>=0.2.2 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (0.4.3)
    Requirement already satisfied: soundfile>=0.10.2 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (0.13.1)
    Requirement already satisfied: pooch>=1.0 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (1.8.2)
    Requirement already satisfied: packaging>=20.0 in ./tflite_env/lib/python3.9/site-packages (from librosa==0.8.1->tflite-model-maker==0.4.2) (25.0)
    Collecting llvmlite<0.37,>=0.36.0rc1 (from numba==0.53->tflite-model-maker==0.4.2)
      Downloading llvmlite-0.36.0-cp39-cp39-manylinux2010_x86_64.whl.metadata (4.9 kB)
    Requirement already satisfied: setuptools in ./tflite_env/lib/python3.9/site-packages (from numba==0.53->tflite-model-maker==0.4.2) (58.1.0)
    Collecting tensorflow>=2.6.0 (from tflite-model-maker==0.4.2)
      Downloading tensorflow-2.8.4-cp39-cp39-manylinux2010_x86_64.whl.metadata (2.9 kB)
    Collecting dataclasses (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading dataclasses-0.6-py3-none-any.whl.metadata (3.0 kB)
    Collecting gin-config (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading gin_config-0.5.0-py3-none-any.whl.metadata (2.9 kB)
    Collecting google-api-python-client>=1.6.7 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_api_python_client-2.169.0-py3-none-any.whl.metadata (6.7 kB)
    Collecting google-cloud-bigquery>=0.31.0 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_cloud_bigquery-3.33.0-py3-none-any.whl.metadata (8.0 kB)
    Collecting kaggle>=1.3.9 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading kaggle-1.7.4.5-py3-none-any.whl.metadata (16 kB)
    Collecting opencv-python-headless (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
    Collecting pandas>=0.22.0 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading pandas-2.2.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (89 kB)
    Requirement already satisfied: psutil>=5.4.3 in ./tflite_env/lib/python3.9/site-packages (from tf-models-official==2.3.0->tflite-model-maker==0.4.2) (7.0.0)
    Collecting py-cpuinfo>=3.3.0 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
    Collecting tf-slim>=1.1.0 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading tf_slim-1.1.0-py2.py3-none-any.whl.metadata (1.6 kB)
    Requirement already satisfied: cycler>=0.10 in ./tflite_env/lib/python3.9/site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker==0.4.2) (0.12.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in ./tflite_env/lib/python3.9/site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker==0.4.2) (1.4.7)
    Requirement already satisfied: pyparsing>=2.2.1 in ./tflite_env/lib/python3.9/site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker==0.4.2) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in ./tflite_env/lib/python3.9/site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker==0.4.2) (2.9.0.post0)
    Requirement already satisfied: astunparse>=1.6.0 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (1.6.3)
    Requirement already satisfied: gast>=0.2.1 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.13.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (1.1.2)
    Requirement already satisfied: libclang>=9.0.1 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (18.1.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.4.0)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.19.6)
    Requirement already satisfied: termcolor>=1.1.0 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.1.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (4.13.2)
    Requirement already satisfied: wrapt>=1.11.0 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (1.17.2)
    Collecting tensorboard<2.9,>=2.8 (from tensorflow>=2.6.0->tflite-model-maker==0.4.2)
      Downloading tensorboard-2.8.0-py3-none-any.whl.metadata (1.9 kB)
    Collecting tensorflow-estimator<2.9,>=2.8 (from tensorflow>=2.6.0->tflite-model-maker==0.4.2)
      Downloading tensorflow_estimator-2.8.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting keras<2.9,>=2.8.0rc0 (from tensorflow>=2.6.0->tflite-model-maker==0.4.2)
      Downloading keras-2.8.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.37.1)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in ./tflite_env/lib/python3.9/site-packages (from tensorflow>=2.6.0->tflite-model-maker==0.4.2) (1.71.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (2.40.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.4.6)
    Requirement already satisfied: markdown>=2.6.8 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.8)
    Requirement already satisfied: requests<3,>=2.21.0 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (2.32.3)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.6.1)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (1.8.1)
    Requirement already satisfied: werkzeug>=0.11.15 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.1.3)
    Requirement already satisfied: wheel>=0.26 in ./tflite_env/lib/python3.9/site-packages (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.45.1)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in ./tflite_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (5.5.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in ./tflite_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.4.2)
    Requirement already satisfied: rsa<5,>=3.1.4 in ./tflite_env/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (4.9.1)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in ./tflite_env/lib/python3.9/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (2.0.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in ./tflite_env/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in ./tflite_env/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.10)
    Requirement already satisfied: certifi>=2017.4.17 in ./tflite_env/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (2025.4.26)
    Requirement already satisfied: pyasn1>=0.1.3 in ./tflite_env/lib/python3.9/site-packages (from rsa<5,>=3.1.4->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (0.6.1)
    Collecting packaging>=20.0 (from librosa==0.8.1->tflite-model-maker==0.4.2)
      Downloading packaging-20.9-py2.py3-none-any.whl.metadata (13 kB)
    Collecting httplib2<1.0.0,>=0.19.0 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading httplib2-0.22.0-py3-none-any.whl.metadata (2.6 kB)
    Collecting google-auth-httplib2<1.0.0,>=0.2.0 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_auth_httplib2-0.2.0-py2.py3-none-any.whl.metadata (2.2 kB)
    Collecting google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0,>=1.31.5 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_api_core-2.24.2-py3-none-any.whl.metadata (3.0 kB)
    Collecting uritemplate<5,>=3.0.1 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading uritemplate-4.1.1-py2.py3-none-any.whl.metadata (2.9 kB)
    Requirement already satisfied: googleapis-common-protos<2.0.0,>=1.56.2 in ./tflite_env/lib/python3.9/site-packages (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2) (1.63.1)
    Collecting proto-plus<2.0.0,>=1.22.3 (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading proto_plus-1.26.1-py3-none-any.whl.metadata (2.2 kB)
    Collecting google-cloud-core<3.0.0,>=2.4.1 (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_cloud_core-2.4.3-py2.py3-none-any.whl.metadata (2.7 kB)
    Collecting google-resumable-media<3.0.0,>=2.0.0 (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_resumable_media-2.7.2-py2.py3-none-any.whl.metadata (2.2 kB)
    INFO: pip is looking at multiple versions of google-cloud-bigquery to determine which version is compatible with other requirements. This could take a while.
    Collecting google-cloud-bigquery>=0.31.0 (from tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_cloud_bigquery-3.31.0-py3-none-any.whl.metadata (7.7 kB)
      Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl.metadata (7.9 kB)
    Collecting google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0,>=1.31.5 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_api_core-2.25.0rc1-py3-none-any.whl.metadata (3.0 kB)
    Collecting grpcio-status<2.0.0,>=1.33.2 (from google-api-core[grpc]<3.0.0dev,>=2.11.1->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading grpcio_status-1.71.0-py3-none-any.whl.metadata (1.1 kB)
    Collecting google-crc32c<2.0dev,>=1.0 (from google-resumable-media<3.0.0,>=2.0.0->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading google_crc32c-1.7.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
    INFO: pip is looking at multiple versions of grpcio-status to determine which version is compatible with other requirements. This could take a while.
    Collecting grpcio-status<2.0.0,>=1.33.2 (from google-api-core[grpc]<3.0.0dev,>=2.11.1->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading grpcio_status-1.70.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.69.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.68.1-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.68.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.67.1-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.67.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.66.2-py3-none-any.whl.metadata (1.1 kB)
    INFO: pip is still looking at multiple versions of grpcio-status to determine which version is compatible with other requirements. This could take a while.
      Downloading grpcio_status-1.66.1-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.66.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.65.5-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.65.4-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.65.2-py3-none-any.whl.metadata (1.1 kB)
    INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
      Downloading grpcio_status-1.65.1-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.64.3-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.64.1-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.64.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.63.2-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.63.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.62.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.62.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.62.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.62.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.61.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.60.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.60.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.60.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.59.5-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.59.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.59.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.59.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.58.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.58.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.57.0-py3-none-any.whl.metadata (1.2 kB)
      Downloading grpcio_status-1.56.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.56.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.55.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.54.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.54.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.54.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.53.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.53.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.53.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.51.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.51.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.50.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.49.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.48.2-py3-none-any.whl.metadata (1.2 kB)
    Collecting bleach (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading bleach-6.2.0-py3-none-any.whl.metadata (30 kB)
    Collecting python-slugify (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading python_slugify-8.0.4-py2.py3-none-any.whl.metadata (8.5 kB)
    Collecting text-unidecode (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading text_unidecode-1.3-py2.py3-none-any.whl.metadata (2.4 kB)
    Requirement already satisfied: tqdm in ./tflite_env/lib/python3.9/site-packages (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker==0.4.2) (4.67.1)
    Collecting webencodings (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in ./tflite_env/lib/python3.9/site-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (8.7.0)
    Requirement already satisfied: zipp>=3.20 in ./tflite_env/lib/python3.9/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.21.0)
    Collecting attrs (from neural-structured-learning>=1.3.1->tflite-model-maker==0.4.2)
      Downloading attrs-25.3.0-py3-none-any.whl.metadata (10 kB)
    Collecting pytz>=2020.1 (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.7 (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker==0.4.2)
      Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: platformdirs>=2.5.0 in ./tflite_env/lib/python3.9/site-packages (from pooch>=1.0->librosa==0.8.1->tflite-model-maker==0.4.2) (4.3.8)
    Requirement already satisfied: oauthlib>=3.0.0 in ./tflite_env/lib/python3.9/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.2.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in ./tflite_env/lib/python3.9/site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->tflite-model-maker==0.4.2) (3.6.0)
    Requirement already satisfied: cffi>=1.0 in ./tflite_env/lib/python3.9/site-packages (from soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker==0.4.2) (1.17.1)
    Requirement already satisfied: pycparser in ./tflite_env/lib/python3.9/site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->tflite-model-maker==0.4.2) (2.22)
    Collecting typeguard<3.0.0,>=2.7 (from tensorflow-addons>=0.11.2->tflite-model-maker==0.4.2)
      Downloading typeguard-2.13.3-py3-none-any.whl.metadata (3.6 kB)
    Requirement already satisfied: click in ./tflite_env/lib/python3.9/site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (8.1.8)
    Requirement already satisfied: dm-tree in ./tflite_env/lib/python3.9/site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (0.1.8)
    Requirement already satisfied: etils>=0.9.0 in ./tflite_env/lib/python3.9/site-packages (from etils[enp,epath]>=0.9.0->tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (1.5.2)
    Requirement already satisfied: promise in ./tflite_env/lib/python3.9/site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (2.3)
    Requirement already satisfied: tensorflow-metadata in ./tflite_env/lib/python3.9/site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (1.13.0)
    Requirement already satisfied: toml in ./tflite_env/lib/python3.9/site-packages (from tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (0.10.2)
    Requirement already satisfied: fsspec in ./tflite_env/lib/python3.9/site-packages (from etils[enp,epath]>=0.9.0->tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (2025.5.0)
    Requirement already satisfied: importlib_resources in ./tflite_env/lib/python3.9/site-packages (from etils[enp,epath]>=0.9.0->tensorflow-datasets>=2.1.0->tflite-model-maker==0.4.2) (6.5.2)
    Requirement already satisfied: sounddevice>=0.4.4 in ./tflite_env/lib/python3.9/site-packages (from tflite-support>=0.4.2->tflite-model-maker==0.4.2) (0.4.5)
    Requirement already satisfied: pybind11>=2.6.0 in ./tflite_env/lib/python3.9/site-packages (from tflite-support>=0.4.2->tflite-model-maker==0.4.2) (2.13.6)
    Requirement already satisfied: MarkupSafe>=2.1.1 in ./tflite_env/lib/python3.9/site-packages (from werkzeug>=0.11.15->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker==0.4.2) (3.0.2)
    Downloading tflite_model_maker-0.4.2-py3-none-any.whl (577 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m577.3/577.3 kB[0m [31m18.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading numba-0.53.0-cp39-cp39-manylinux2014_x86_64.whl (3.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.4/3.4 MB[0m [31m80.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading scann-1.2.6-cp39-cp39-manylinux2014_x86_64.whl (10.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.9/10.9 MB[0m [31m137.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tf_models_official-2.3.0-py2.py3-none-any.whl (840 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m840.9/840.9 kB[0m [31m36.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading llvmlite-0.36.0-cp39-cp39-manylinux2010_x86_64.whl (25.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m25.3/25.3 MB[0m [31m95.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading matplotlib-3.4.3-cp39-cp39-manylinux1_x86_64.whl (10.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.3/10.3 MB[0m [31m140.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflow-2.8.4-cp39-cp39-manylinux2010_x86_64.whl (498.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m498.1/498.1 MB[0m [31m49.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading keras-2.8.0-py2.py3-none-any.whl (1.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 MB[0m [31m34.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorboard-2.8.0-py3-none-any.whl (5.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.8/5.8 MB[0m [31m78.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflow_estimator-2.8.0-py2.py3-none-any.whl (462 kB)
    Downloading tensorflowjs-3.18.0-py3-none-any.whl (77 kB)
    Downloading packaging-20.9-py2.py3-none-any.whl (40 kB)
    Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    Downloading cython-3.1.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.3/3.3 MB[0m [31m69.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading google_api_python_client-2.169.0-py3-none-any.whl (13.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.3/13.3 MB[0m [31m75.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading google_auth_httplib2-0.2.0-py2.py3-none-any.whl (9.3 kB)
    Downloading httplib2-0.22.0-py3-none-any.whl (96 kB)
    Downloading proto_plus-1.26.1-py3-none-any.whl (50 kB)
    Downloading uritemplate-4.1.1-py2.py3-none-any.whl (10 kB)
    Downloading google_cloud_bigquery-3.30.0-py2.py3-none-any.whl (247 kB)
    Downloading google_api_core-2.25.0rc1-py3-none-any.whl (160 kB)
    Downloading google_cloud_core-2.4.3-py2.py3-none-any.whl (29 kB)
    Downloading google_resumable_media-2.7.2-py2.py3-none-any.whl (81 kB)
    Downloading google_crc32c-1.7.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37 kB)
    Downloading grpcio_status-1.48.2-py3-none-any.whl (14 kB)
    Downloading kaggle-1.7.4.5-py3-none-any.whl (181 kB)
    Downloading lxml-5.4.0-cp39-cp39-manylinux_2_28_x86_64.whl (5.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.1/5.1 MB[0m [31m79.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading neural_structured_learning-1.4.0-py2.py3-none-any.whl (128 kB)
    Downloading pandas-2.2.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.1/13.1 MB[0m [31m69.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
    Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
    Downloading PyYAML-6.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (737 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m737.4/737.4 kB[0m [31m24.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflow_addons-0.23.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (611 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m611.8/611.8 kB[0m [31m25.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading typeguard-2.13.3-py3-none-any.whl (17 kB)
    Downloading tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
    Downloading tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
    Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
    Downloading attrs-25.3.0-py3-none-any.whl (63 kB)
    Downloading bleach-6.2.0-py3-none-any.whl (163 kB)
    Downloading dataclasses-0.6-py3-none-any.whl (14 kB)
    Downloading gin_config-0.5.0-py3-none-any.whl (61 kB)
    Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.0/50.0 MB[0m [31m39.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading python_slugify-8.0.4-py2.py3-none-any.whl (10 kB)
    Downloading text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
    Downloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
    Building wheels for collected packages: fire
    [33m  DEPRECATION: Building 'fire' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'fire'. Discussion can be found at https://github.com/pypa/pip/issues/6334[0m[33m
    [0m  Building wheel for fire (setup.py) ... [?25l[?25hdone
      Created wheel for fire: filename=fire-0.7.0-py3-none-any.whl size=114261 sha256=ada8aa4d48c0bedf481110530a70b2fe0f4ce92b5c25c8b48b9ae6704ede5ee8
      Stored in directory: /root/.cache/pip/wheels/3b/ee/ac/319a7b7f331f61050d0d54425079b2a883b445be3c7284a4eb
    Successfully built fire
    Installing collected packages: webencodings, text-unidecode, tensorflow-estimator, pytz, py-cpuinfo, keras, gin-config, dataclasses, urllib3, uritemplate, tzdata, typeguard, tf-slim, tensorflow-model-optimization, PyYAML, python-slugify, proto-plus, packaging, opencv-python-headless, lxml, llvmlite, httplib2, google-crc32c, fire, Cython, bleach, attrs, tensorflow-addons, pandas, numba, neural-structured-learning, matplotlib, grpcio-status, google-resumable-media, kaggle, google-auth-httplib2, google-api-core, google-cloud-core, google-api-python-client, tensorboard, google-cloud-bigquery, tensorflow, tf-models-official, tensorflowjs, scann, tflite-model-maker
    [2K  Attempting uninstall: tensorflow-estimator
    [2K    Found existing installation: tensorflow-estimator 2.10.0
    [2K    Uninstalling tensorflow-estimator-2.10.0:
    [2K      Successfully uninstalled tensorflow-estimator-2.10.0
    [2K  Attempting uninstall: keras
    [2K    Found existing installation: keras 2.10.0
    [2K    Uninstalling keras-2.10.0:
    [2K      Successfully uninstalled keras-2.10.0
    [2K  Attempting uninstall: urllib3
    [2K    Found existing installation: urllib3 2.4.0
    [2K    Uninstalling urllib3-2.4.0:
    [2K      Successfully uninstalled urllib3-2.4.0
    [2K  Attempting uninstall: packaging
    [2K    Found existing installation: packaging 25.0
    [2K    Uninstalling packaging-25.0:
    [2K      Successfully uninstalled packaging-25.0
    [2K  Attempting uninstall: llvmlite
    [2K    Found existing installation: llvmlite 0.43.0
    [2K    Uninstalling llvmlite-0.43.0:
    [2K      Successfully uninstalled llvmlite-0.43.0
    [2K  Attempting uninstall: numba
    [2K    Found existing installation: numba 0.60.0
    [2K    Uninstalling numba-0.60.0:
    [2K      Successfully uninstalled numba-0.60.0
    [2K  Attempting uninstall: matplotlib
    [2K    Found existing installation: matplotlib 3.5.3
    [2K    Uninstalling matplotlib-3.5.3:
    [2K      Successfully uninstalled matplotlib-3.5.3
    [2K  Attempting uninstall: tensorboard
    [2K    Found existing installation: tensorboard 2.10.1
    [2K    Uninstalling tensorboard-2.10.1:
    [2K      Successfully uninstalled tensorboard-2.10.1
    [2K  Attempting uninstall: tensorflow
    [2K    Found existing installation: tensorflow 2.10.0
    [2K    Uninstalling tensorflow-2.10.0:
    [2K      Successfully uninstalled tensorflow-2.10.0
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46/46[0m [tflite-model-maker]
    [1A[2KSuccessfully installed Cython-3.1.1 PyYAML-6.0.2 attrs-25.3.0 bleach-6.2.0 dataclasses-0.6 fire-0.7.0 gin-config-0.5.0 google-api-core-2.25.0rc1 google-api-python-client-2.169.0 google-auth-httplib2-0.2.0 google-cloud-bigquery-3.30.0 google-cloud-core-2.4.3 google-crc32c-1.7.1 google-resumable-media-2.7.2 grpcio-status-1.48.2 httplib2-0.22.0 kaggle-1.7.4.5 keras-2.8.0 llvmlite-0.36.0 lxml-5.4.0 matplotlib-3.4.3 neural-structured-learning-1.4.0 numba-0.53.0 opencv-python-headless-4.11.0.86 packaging-20.9 pandas-2.2.3 proto-plus-1.26.1 py-cpuinfo-9.0.0 python-slugify-8.0.4 pytz-2025.2 scann-1.2.6 tensorboard-2.8.0 tensorflow-2.8.4 tensorflow-addons-0.23.0 tensorflow-estimator-2.8.0 tensorflow-model-optimization-0.8.0 tensorflowjs-3.18.0 text-unidecode-1.3 tf-models-official-2.3.0 tf-slim-1.1.0 tflite-model-maker-0.4.2 typeguard-2.13.3 tzdata-2025.2 uritemplate-4.1.1 urllib3-1.25.11 webencodings-0.5.1


# 补充缺失依赖


```python
! /content/tflite_env/bin/pip install matplotlib_inline IPython


```

    Collecting matplotlib_inline
      Downloading matplotlib_inline-0.1.7-py3-none-any.whl.metadata (3.9 kB)
    Collecting IPython
      Downloading ipython-8.18.1-py3-none-any.whl.metadata (6.0 kB)
    Collecting traitlets (from matplotlib_inline)
      Downloading traitlets-5.14.3-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: decorator in ./tflite_env/lib/python3.9/site-packages (from IPython) (5.2.1)
    Collecting jedi>=0.16 (from IPython)
      Downloading jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)
    Collecting prompt-toolkit<3.1.0,>=3.0.41 (from IPython)
      Downloading prompt_toolkit-3.0.51-py3-none-any.whl.metadata (6.4 kB)
    Collecting pygments>=2.4.0 (from IPython)
      Downloading pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)
    Collecting stack-data (from IPython)
      Downloading stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)
    Requirement already satisfied: typing-extensions in ./tflite_env/lib/python3.9/site-packages (from IPython) (4.13.2)
    Collecting exceptiongroup (from IPython)
      Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
    Collecting pexpect>4.3 (from IPython)
      Downloading pexpect-4.9.0-py2.py3-none-any.whl.metadata (2.5 kB)
    Collecting wcwidth (from prompt-toolkit<3.1.0,>=3.0.41->IPython)
      Downloading wcwidth-0.2.13-py2.py3-none-any.whl.metadata (14 kB)
    Collecting parso<0.9.0,>=0.8.4 (from jedi>=0.16->IPython)
      Downloading parso-0.8.4-py2.py3-none-any.whl.metadata (7.7 kB)
    Collecting ptyprocess>=0.5 (from pexpect>4.3->IPython)
      Downloading ptyprocess-0.7.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting executing>=1.2.0 (from stack-data->IPython)
      Downloading executing-2.2.0-py2.py3-none-any.whl.metadata (8.9 kB)
    Collecting asttokens>=2.1.0 (from stack-data->IPython)
      Downloading asttokens-3.0.0-py3-none-any.whl.metadata (4.7 kB)
    Collecting pure-eval (from stack-data->IPython)
      Downloading pure_eval-0.2.3-py3-none-any.whl.metadata (6.3 kB)
    Downloading matplotlib_inline-0.1.7-py3-none-any.whl (9.9 kB)
    Downloading ipython-8.18.1-py3-none-any.whl (808 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m808.2/808.2 kB[0m [31m18.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading prompt_toolkit-3.0.51-py3-none-any.whl (387 kB)
    Downloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m46.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading parso-0.8.4-py2.py3-none-any.whl (103 kB)
    Downloading pexpect-4.9.0-py2.py3-none-any.whl (63 kB)
    Downloading ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
    Downloading pygments-2.19.1-py3-none-any.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m43.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading traitlets-5.14.3-py3-none-any.whl (85 kB)
    Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
    Downloading stack_data-0.6.3-py3-none-any.whl (24 kB)
    Downloading asttokens-3.0.0-py3-none-any.whl (26 kB)
    Downloading executing-2.2.0-py2.py3-none-any.whl (26 kB)
    Downloading pure_eval-0.2.3-py3-none-any.whl (11 kB)
    Downloading wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)
    Installing collected packages: wcwidth, pure-eval, ptyprocess, traitlets, pygments, prompt-toolkit, pexpect, parso, executing, exceptiongroup, asttokens, stack-data, matplotlib_inline, jedi, IPython
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15/15[0m [IPython]
    [1A[2KSuccessfully installed IPython-8.18.1 asttokens-3.0.0 exceptiongroup-1.3.0 executing-2.2.0 jedi-0.19.2 matplotlib_inline-0.1.7 parso-0.8.4 pexpect-4.9.0 prompt-toolkit-3.0.51 ptyprocess-0.7.0 pure-eval-0.2.3 pygments-2.19.1 stack-data-0.6.3 traitlets-5.14.3 wcwidth-0.2.13


# 验证是否成功安装


```python
! /content/tflite_env/bin/python -c "from tflite_model_maker import image_classifier; print('TFLite Model Maker 已成功导入')"

```

    2025-05-21 01:01:48.212516: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64
    2025-05-21 01:01:48.212570: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    /content/tflite_env/lib/python3.9/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 
    
    TensorFlow Addons (TFA) has ended development and introduction of new features.
    TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
    Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 
    
    For more information see: https://github.com/tensorflow/addons/issues/2807 
    
      warnings.warn(
    /content/tflite_env/lib/python3.9/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.13.0 and strictly below 2.16.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    TFLite Model Maker 已成功导入


# 训练模型（花卉分类）


```python
# step_train.py
with open('/content/step_train.py', 'w') as f:
    f.write("""
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

image_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)
loss, acc = model.evaluate(test_data)
print(f'✅ 测试准确率: {acc:.4f}')
model.export(export_dir='.')
""")
! /content/tflite_env/bin/python /content/step_train.py


```

    2025-05-21 01:05:47.632644: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64
    2025-05-21 01:05:47.632686: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    /content/tflite_env/lib/python3.9/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 
    
    TensorFlow Addons (TFA) has ended development and introduction of new features.
    TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
    Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 
    
    For more information see: https://github.com/tensorflow/addons/issues/2807 
    
      warnings.warn(
    /content/tflite_env/lib/python3.9/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.13.0 and strictly below 2.16.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228818944/228813984 [==============================] - 2s 0us/step
    228827136/228813984 [==============================] - 2s 0us/step
    2025-05-21 01:06:01.890589: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/nvidia/lib:/usr/local/nvidia/lib64
    2025-05-21 01:06:01.890656: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2025-05-21 01:06:01.890697: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (87048b14c4f2): /proc/driver/nvidia/version does not exist
    2025-05-21 01:06:01.891384: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    103/103 [==============================] - 176s 2s/step - loss: 0.8781 - accuracy: 0.7561
    Epoch 2/5
    103/103 [==============================] - 171s 2s/step - loss: 0.6481 - accuracy: 0.8935
    Epoch 3/5
    103/103 [==============================] - 178s 2s/step - loss: 0.6120 - accuracy: 0.9229
    Epoch 4/5
    103/103 [==============================] - 171s 2s/step - loss: 0.5968 - accuracy: 0.9293
    Epoch 5/5
    103/103 [==============================] - 168s 2s/step - loss: 0.5833 - accuracy: 0.9345
    12/12 [==============================] - 23s 1s/step - loss: 0.6566 - accuracy: 0.9046
    ✅ 测试准确率: 0.9046
    2025-05-21 01:22:27.007962: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    2025-05-21 01:22:33.599584: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
    2025-05-21 01:22:33.599790: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
    2025-05-21 01:22:33.673800: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
      function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 42.421ms.
      function_optimizer: function_optimizer did nothing. time = 0.021ms.
    
    /content/tflite_env/lib/python3.9/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    2025-05-21 01:22:34.782040: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
    2025-05-21 01:22:34.782106: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
    fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3


# tflite模型下载


```python
from google.colab import files
files.download('model.tflite')

```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>

