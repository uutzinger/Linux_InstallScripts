# Install CNN Frameworks

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
#
mkdir build
cd build
# build 64-bit ncnn
cmake ..
make -j$(nproc)
make install
# copy output to dirs
sudo mkdir /usr/local/lib/ncnn
sudo cp -r install/include/ncnn /usr/local/include/ncnn
sudo cp -r install/lib/libncnn.a /usr/local/lib/ncnn/libncnn.a
```

Python: remove opencv requirement from requirements in setup.py and python/requirements.txt otherwise our custom opencv installation will be overwritten.

```
cd ~/Build/ncnn
git submodule update --init
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
cmake --build . -j$(nproc)
cmake --build . -j -t benchmark_model
cmake --build . -j -t label_image

sudo make install
sudo ldconfig
```

Python

```
PYTHON=python3 ../tensorflow/tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
```

## Tensorflow
