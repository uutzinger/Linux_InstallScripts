# Install OpenCV and CNN Frameworks

## Prepare

```
# check for updates
sudo apt-get update
sudo apt-get upgrade
# install dependencies
sudo apt-get install -y cmake wget git unzip pkg-config
sudo apt-get install -y build-essential gcc g++
sudo apt-get install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo apt-get install -y libprotobuf-dev protobuf-compiler
sudo apt-get install -y libjpeg-dev libtiff-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk* libgtk-3-dev
sudo apt-get install -y libgstreamer1.0-dev gstreamer1.0-gtk3
sudo apt-get install -y libgstreamer-plugins-base1.0-dev gstreamer1.0-gl
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y python3-dev python3-numpy python3-pip
sudo apt-get install -y libtbbmalloc2 libtbb-dev
sudo apt-get install -y libv4l-dev v4l-utils
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y liblapack-dev gfortran libhdf5-dev
sudo apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt-get install -y libtbbmalloc2 libtbb-dev
sudo apt-get install -y qtbase5-dev
sudo apt-get install -y flatbuffers-compiler
```

## Opencv

Change the swap memory size.

```
# enlarge the boundary (CONF_MAXSWAP) from 2048 to 4096
sudo nano /sbin/dphys-swapfile
# give the required memory size (CONF_SWAPSIZE) from 100 to 4096
sudo nano /etc/dphys-swapfile
# reboot afterwards
sudo reboot
```

Clone the respository.

```
cd ~
git clone --depth=1 https://github.com/opencv/opencv.git
git clone --depth=1 https://github.com/opencv/opencv_contrib.git

cd ~/opencv
mkdir build
cd build


$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D ENABLE_NEON=ON \
-D WITH_OPENMP=ON \
-D WITH_OPENCL=OFF \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=OFF \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D WITH_VTK=OFF \
-D WITH_QT=ON \
-D WITH_PROTOBUF=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF ..

make -j4
sudo make install
sudo ldconfig
```

Test

```
python
import cv2
print( cv2.getBuildInformation() )
```

## pyTorch

```
pip3 install setuptools numpy Cython
pip3 install requests
# install PyTorch and Torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# you may like to install Torchaudio also
pip3 install torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## NCNN

```
# download ncnn
git clone --depth=1 https://github.com/Tencent/ncnn.git
# install ncnn
cd ncnn
mkdir build
cd build
# build 64-bit ncnn
cmake -D NCNN_DISABLE_RTTI=OFF -D NCNN_BUILD_TOOLS=ON \
-D CMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j4
make install
# copy output to dirs
sudo mkdir /usr/local/lib/ncnn
sudo cp -r install/include/ncnn /usr/local/include/ncnn
sudo cp -r install/lib/libncnn.a /usr/local/lib/ncnn/libncnn.a
```

Python: remove opencv requirement from requirements in setup.py and python/requirements.txt

```
python3 -m setup build
sudo python3 -m setup install
```

## MNN

```
git clone --depth=1 https://github.com/alibaba/MNN.git
# common preparation (installing the flatbuffers)
cd MNN
./schema/generate.sh
# install MNN
mkdir build
cd build
# generate build script
cmake -D CMAKE_BUILD_TYPE=Release \
        -D MNN_OPENMP=ON \
        -D MNN_USE_THREAD_POOL=OFF \
        -D MNN_BUILD_QUANTOOLS=ON \
        -D MNN_BUILD_CONVERTER=ON \
        -D MNN_BUILD_DEMO=ON \
        -D MNN_BUILD_BENCHMARK=ON ..

# build
make -j4
sudo make install

# get some models
cd ~/MNN
./tools/script/get_model.sh

# build and install python package
cd pymnn/pip_package
python3 build_deps.py
sudo python3 setup.py install

```

## Tensorflow lite

```
git clone https://github.com/tensorflow/tensorflow.git tensorflow
mkdir tflite_build
cd tflite_build
cmake ../tensorflow/tensorflow/lite
cmake --build . -j4
cmake --build . -j -t benchmark_model
cmake --build . -j -t label_image

sudo make install
sudo ldconfig
```

Python

```
PYTHON=python3 ../tensorflow/tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```
