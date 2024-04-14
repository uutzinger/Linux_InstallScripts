# OpenCV with CUDA12 on POP_OS

## Table of Content

- [OpenCV with CUDA12 on POP_OS](#opencv-with-cuda12-on-pop-os)
  * [Table of Content](#table-of-content)
  * [References](#references)
  * [Update](#update)
  * [Dependencies](#dependencies)
  * [Install Packages for CUDA support](#install-packages-for-cuda-support)
  * [Eigen](#eigen)
  * [NVIDIA Video Codec SDK](#nvidia-video-codec-sdk)
  * [HALIDE](#halide)
  * [Obtain OpenCV Sources](#obtain-opencv-sources)
  * [Configure](#configure)
  * [Building](#building)
  * [Installing](#installing)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## References
- [1 CUDA12 PopOS](https://toranbillups.com/blog/archive/2023/08/19/install-cuda-12-on-popos/)
- [2 OpenCV](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html)
- [3 OpenCV Options](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html)
- [4 OpenCV CUDA](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
- [5 OpenCV on Jetson](https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/blob/main/OpenCV-4-9-0.sh)

## Update
```
sudo apt update
sudo apt upgrade
```
## Dependencies
```
# build
sudo apt install -y build-essential git unzip pkg-config yasm cmake checkinstall zlib1g-dev doxygen

# python
sudo apt install -y python3-dev python3-numpy python3-pip python3-testresources
# sudo apt install -y python-dev python-numpy python-pip

# gstreamer
sudo apt install -y libgstreamer1.0-dev gstreamer1.0-tools libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev

# display
sudo apt install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
sudo apt install -y libvtk9-dev


# parallel
sudo apt install -y libtbb-dev

# linear algebra
sudo apt install -y libopenblas-dev libatlas-base-dev libblas-dev gfortran
sudo apt install -y liblapack-dev liblapacke-dev libeigen3-dev

# buffers
sudo apt install -y libprotobuf-dev protobuf-compiler
sudo apt install -y libgoogle-glog-dev libgflags-dev

sudo apt install -y libgphoto2-dev
sudo apt install -y libhdf5-dev 

# graphics and sound
sudo apt install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev
sudo apt install -y libpng-dev libtiff-dev libglew-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libdc1394-dev libxine2-dev
sudo apt install -y libv4l-dev v4l-utils qv4l2
sudo apt install -y libtesseract-dev libpostproc-dev
sudo apt install -y libvorbis-dev
sudo apt install -y libva-dev
sudo apt install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt install -y libopencore-amrnb-dev libopencore-amrwb-dev

# Camera programming interface libs
sudo ln -s -f ../libv4l1-videodev.h /usr/include/linux/videodev.h
```

## Install Packages for CUDA support

We should have the NVIDIA graphics drivers. Latest version for Pop OS is: `libnvidia-compute-550` . You can check with `sudo synapic`.

You should not have the system76-cuda and system76-cudnn drivers as we will get them from NVIDIA. This makes installation more complicated but the packages and instructions on system76 do not work. You want to remove the following packages from your system if they are present.

```
system76-cuda
system76-cuda-latest
system76-cuda-11.2
system76-cudnn-11.2
cuda-toolkit-*
```
Now we need to select the appropriate version. The nvcc compiler has a maximum supported gnu c compiler (GCC) version.  https://gist.github.com/ax3l/9489132 lists them as:
```
With CUDA 11.2   its gcc 10. 
With CUDA 12     its gcc 12.1
With CUDA 12.1-3 its gcc 12.2
With CUDA 12.4   its gcc 13.2. 
```

We will want to install the latest CUDA toolkit that is supported by the gcc version available on our system which you can find with `apt list gcc-1*`. It is gcc-12.3 on 22.04 Pop OS, therefore CUDA 12.3 is the latest we can use.

We will want to download from the NVIDIA repository. Any other attempt did not work for me. Thanks for reference [1].

```
# Prioritize NVIDIA packages
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Fetch keys
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Add repository (there might be geo restrictions but you can google for solution)
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# Refresh and install CUDA, choose correct CUDA version
sudo apt update
sudo apt install cuda-toolkit-12-3
```

Next we want to download CuDNN from https://developer.nvidia.com/cudnn.  We can check on nividia website for [manual download](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local). This will tell us what the latest version is. 

We can just use wget on that file as shown here:

```
wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.debsudo 
dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.debsudo 
sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-local-52C3CBCA-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

Now we update bash with `nano ~./bashrc` to use the installed CUDA libraries.
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
```

## Eigen
This provies some matrix algebra support. OpenCV install script did not find libeigen3-dev for me.

```
wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip Eigen.zip 
sudo cp -r eigen-3.4.0/Eigen /usr/local/include 
```

## NVIDIA Video Codec SDK
This provies CUDA video encoder and decoder support. It compiles but I have not found a test program that shows that this works.

Obtain package from https://developer.nvidia.com/nvidia-video-codec-sdk/download
This requires a developer account and NVIDIA.

Exdtract and
```
cd to_videocodecsdk_folder
sudo cp Interface/*.h  /usr/local/cuda/include
sudo cp Lib/linux/stubs/x86_64/*.so /usr/local/cuda/lib64
```

## HALIDE
Halide is a compiler extension for signal processing acceleration. OpenCV supports Halide.

Halide requires LLVM, so lets get it.

```
cd ~/build
git clone --depth 1 --branch main https://github.com/llvm/llvm-project.git
cmake -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
    -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Hexagon;WebAssembly;RISCV" \
    -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -S llvm-project/llvm -B llvm-build
cmake --build llvm-build -j$(nproc)
cmake --install llvm-build --prefix llvm-install

sudo nano ~/.bashrc

export LLVM_ROOT=~/build/llvm-install
export LLVM_CONFIG=$LLVM_ROOT/bin/llvm-config
```

Now we can build Halide and install it.
```
cd ~/build
git clone https://github.com/halide/Halide.git
cd Halide
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=$LLVM_ROOT/lib/cmake/llvm -S . -B build
cmake --build build
sudo cmake --install build
```

## Obtain OpenCV Sources
```
cd ~
mkdir Build
cd Build
git clone https://github.com/opencv/opencv.git opencv
git clone https://github.com/opencv/opencv_contrib.git opencv_contrib
git clone https://github.com/opencv/opencv_extra.git opencv_extra

cd ~/Build/opencv
mkdir build
cd build
```

## Configure

### STEP 1

Check configuration with `cmake-gui ..` and choose Unix Makefiles or Ninja. With the default settings, hit "Configure" and "Generate" which should generate cmake files with no errors. Make sure this minimal configuration compiles with `make -j $(nproc)` or `ninja -j $(nproc)` without errors.

### STEP 2

- Lets set the EXTRAS folder to the donwloaded opencv_contrib/modules. 

The following settings were turned on and worked for me:

```
WITH_...
1394, ADE, EIGEN, FFMPEG, FLATBUFFERS, GSTREAMER, IMGCODEC_HDR,PFM,PXM,SUN, IPP, ITT, JASPER, JPEG, LAPACK, OBSENSOR, OPENCL,AMDBLAS, OPENXR, OPENJPG, PNG, PROTOBUF, PTHREADS_PF, TESSERACT, TIFF, V4L, VA, VA_INTEL, VTK, WEPB
```

- I added `OPENCV_ENABLE_NONFREE=ON` and `BUILD_EXAMPLES=ON` and choose to install the examples: `INSTALL_BIN_EXAMPLES=ON` `INSTALL_C_EXAMPLES=ON` `INSTALL_PYTHON_EXAMPLES=ON`

- I added `WITH_HALIDE=ON` and `WITH_QT=ON` and I enabled `IPP_ALL`.

To save time, you can skip a build here and continue with step 3.

### STEP 3

Let's enable `WITH_CUDA=ON` in cmage-gui. 
If this issue is still open, you need to apply the modifications https://github.com/opencv/opencv_contrib/issues/3711

After turning on WITH_CUDA, additional settings became available and were automnatically selected: `WITH_... CUBLAS, CUDNN, CUFFT, NVCUVENC, NVCUVID`

We can choose CUDA_ARCH_PTX but the default setting will be ok: According to https://en.wikipedia.org/wiki/CUDA CUDA 12.3 supports ARCH_BIN 5.0 to 9.0.  `nvidia-smi` shows driver version (550.67) supporting CUDA 12.4.  Based on https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html this will work because CUDA 12.3 requires >=545.23.06. https://en.wikipedia.org/wiki/CUDA shows that my RTX3060 GPU is architecture 8.6. I did NOT change the default 9.0 to 8.6.

## Building

Usually ninja or make can figure out which components need updating 
`make clean` or `ninja -t clean`
`make -j $(npoc)` or `ninja -j $(npoc)`

## Installing

```
sudo make install
```

Your `/etc/ld.so.conf.d/opencv.conf' should state `/usr/local/cuda/lib64` otherwise ```sudo /bin/bash -c 'echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/opencv.conf'```

```
sudo ldconfig
```

The file `/usr/lib/python3/dist-packages/cv2/config-3.10.py` should look like

``` 
    PYTHON_EXTENSIONS_PATHS = [
    os.path.join('/usr/lib/python3/dist-packages/cv2', 'python-3.10')
    ] + PYTHON_EXTENSIONS_PATHS
``` 

Check that `cv2` loads in python:

```
python 3
import cv2
cv2.__version__
print(cv2.getBuildInformation())
```

## Running a Python Test

This compares CUDA matrix multiplication with numpy and general multiplications.

```
import numpy as np
import cv2
import time

npTmp = np.random.random((1024, 1024)).astype(np.float32)
npMat1 = np.stack([npTmp,npTmp],axis=2)
npMat2 = npMat1
npMat3 = npTmp + npTmp*1j
npMat4 = npMat3
cuMat1 = cv2.cuda_GpuMat()
cuMat2 = cv2.cuda_GpuMat()
cuMat1.upload(npMat1)
cuMat2.upload(npMat2)

jit_time = time.time()
_ = cv2.cuda.gemm(cuMat1, cuMat2,1,None,0,None,1)
current_time = time.time()

for i in range(100):
   _ = cv2.cuda.gemm(cuMat1, cuMat2,1,None,0,None,1)

cuda_time = time.time()

for i in range(100):
   _ = cv2.gemm(npMat1,npMat2,1,None,0,None,1)

cpu_time = time.time()

for i in range(100):
   _ = npMat3 @ npMat4

np_time = time.time()

# CUDA jit compilation
print('CUDA jit compilation time is   : {}'.format((current_time-jit_time)))

# CUDA time
print('CUDA Matrix Multiplication time is   : {}'.format((cuda_time-current_time)/100.0))

# OpenCV Mat Pultiplication
print('OpenCV Matrix Multiplication time is : {}'.format((cpu_time-cuda_time)/100.0))

# NumPy Mat Multiplication
print('NumPy  Matrix Multiplication  time is  : {}'.format((np_time-cpu_time)/100.0))
```
