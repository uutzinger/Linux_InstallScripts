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
sudo apt install -y build-essential git unzip pkg-config yasm cmake checkinstall zlib1g-dev
sudo apt install -y python3-dev python3-numpy python3-pip python3-testresources
# sudo apt install -y python-dev python-numpy python-pip
sudo apt install -y libgstreamer1.0-dev gstreamer1.0-tools libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev
sudo apt install -y libgtk-3-dev
sudo apt install -y libtbb-dev
sudo apt install -y libprotobuf-dev protobuf-compiler
sudo apt install -y libgoogle-glog-dev libgflags-dev
sudo apt install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
sudo apt install -y libopenblas-dev libatlas-base-dev libblas-dev gfortran
sudo apt install -y liblapack-dev liblapacke-dev libeigen3-dev
sudo apt install -y libhdf5-dev libprotobuf-dev protobuf-compiler
sudo apt install -y libgoogle-glog-dev libgflags-dev

# Common
sudo apt install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev
sudo apt install -y libpng-dev libtiff-dev libglew-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libdc1394-dev libxine2-dev
sudo apt install -y libv4l-dev v4l-utils qv4l2
sudo apt install -y libtesseract-dev libpostproc-dev
sudo apt install -y libvorbis-dev
sudo apt install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt install -y libopencore-amrnb-dev libopencore-amrwb-dev

# Camera programming interface libs
sudo ln -s -f ../libv4l1-videodev.h /usr/include/linux/videodev.h
```

## Install Packages for CUDA support

We should have the following drivers:

```
libnvidia-compute-550
```
or newer and not the system76 cuda and cudnn drivers:

You might need to remove the system76 dirvers with synaptic:

```
system76-cuda
system76-cuda-latest
system76-cuda-11.2
system76-cudnn-11.2
```

The nvcc compiler has a max supported GCC version.  https://gist.github.com/ax3l/9489132
```
With CUDA 11.2   its gcc 10. 
With CUDA 12     its gcc 12.1
With CUDA 12.1-3 its gcc 12.2
With CUDA 12.4   its gcc 13.2. 
```

We will want to install the latest CUDA toolkit that is supported by the gcc version available on your system which you can find with `apt list gcc-1*`. It is gcc-12 on 22.04 Pop OS, therefore CUDA 12.3 is the latest we can use.

We will want to download from the NVIDIA repository.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
# here you need to choose the CUDA version
sudo apt install cuda-toolkit-12-3
```

Next we want to download CuDNN from https://developer.nvidia.com/cudnn.  We can check on nividia website for [manual download](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) directly what the latest version is or we can just use wget.

```
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install cudnn
```

Now we update bash with `nano ~./bashrc` to use the installed CUDA libraries.
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
```

## Eigen
This provies some matrix algebra support.

```
wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip Eigen.zip 
sudo cp -r eigen-3.4.0/Eigen /usr/local/include 
```

## NVIDIA Video Codec SDK
This provies CUDA video encoder and decoder support.

Obtain package from https://developer.nvidia.com/nvidia-video-codec-sdk/download

This requires a developer account and NVIDIA.

Exdtract and
```
cd tovideocodecsdk_folder
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

Check configuration with `cmake-gui ..` and choose Unix Makefiles. With the default settings, hit "Configure" and "Generate" which should generate cmake files with no errors. Make sure this minimal configuration compiles with `make -j $(nproc)` without errors.

Let's enable CUDA by searching for for CUDA and enable WITH_CUDA in cmage-gui. Also set EXTRAS folder to where you donwloaded opencv_contrib/modules. 

If this issue is still open, you need to apply the modificaations https://github.com/opencv/opencv_contrib/issues/3711

If CUDA configuration completes we can finally set a few more options:

```
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D CPU_BASELINE=SSE3 \
-D WITH_OPENCL=ON \
-D OPENCV_IPP_ENABLE_ALL=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_EIGEN=ON \
-D ENABLE_FAST_MATH=OFF \
-D CUDA_FAST_MATH=OFF \
-D OPENCV_DNN_CUDA=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=OFF \
-D WITH_OPENMP=OFF \
-D WITH_HALIDE=ON \
-D WITH_QT=ON \
-D WITH_GTK=ON \
-D WITH_OPENGL=OFF \
-D BUILD_TIFF=OFF \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_1394=ON \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_opencv_apps=ON \
-D WITH_V4L=ON \
-D WITH_PROTOBUF=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_opencv_cudacodec=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON ..
```

## Building

```
make -j $(npoc) 
```

## Installing

```
sudo make install
sudo /bin/bash -c 'echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
```