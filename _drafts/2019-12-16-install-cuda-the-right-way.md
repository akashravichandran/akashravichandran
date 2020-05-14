---
layout: post
title: Installing CUDA the right way for beginners! (Ubuntu 18.04 (CUDA 10))
categories: Installation
comments: False
---


## Why this Blog Post?
To help beginners install cuda the right way and understand what goes under the installation process.

## What is Cuda

CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit for general purpose processing — an approach termed GPU.


## How to install CUDA the right way?
Essentially to install CUDA the right way, you will need only three components.

- CUDA Installer - This is the installation file which enables CUDA in our local directory and make sure it is accessible for any operation that requires GPU power. Make sure to install the compatible CUDA for the Nvidia driver installed.

<!--more-->

- Nvidia Driver - The correct version
Make sure to have a look and install the compatible Nvidia Driver and CUDA version. Reference Link - https://docs.nvidia.com/deploy/cuda-compatibility/

- CuDNN development and runtime libraries - The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.





## To verify whether CUDA is compatible
```
lspci | grep -i nvidia
```
If you get an output in terminal which contains NVIDIA in VGA devices, then you are ready to install CUDA.

Verifying the Ubuntu version 
```
uname -m && cat /etc/*release
```
<!--more-->

## Steps Involved
[0. Removing Previous or Old Versions of Cuda](##step-0:-removing-previous-or-old-versions-of-cuda)

[1. Add NVIDIA package repositories](##step-1:-add-nvidia-package-repositories)

[2. Install NVIDIA driver](##step-2:install-nvidia-driver)

[3. Install development and runtime libraries](##step-3:-install-development-and-runtime-libraries)

[4. Install Tensorflow and check!](##step-4:-install-tensorflow-and-check!)

[Things to Keep in mind](##-things-to-keep-in-mind)

[References](##references)

## Step 0: Removing Previous or Old Versions of Cuda
```
sudo apt-get remove --purge nvidia*

sudo apt-get autoremove
```

## Step 1: Add NVIDIA package repositories

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt-get update

```

In the above series of steps, we are downloading and installing the respective NVIDIA packages.

If you want to download and install another version of CUDA, kindly access the below two links and change the **wget** commands accordingly. 

```
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/

http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/
```

## Step 2: Install NVIDIA driver
```
sudo apt-get install --no-install-recommends nvidia-driver-418
```
Reboot. Check that GPUs are visible using the command: **nvidia-smi**

### Nvidia Driver Installation Alternative method - Incase you have a Ubuntu Desktop installed and not a Ubuntu Server
You can install the Ubuntu driver in the following way
- Systems Settings 
- Softwares & Updates 
- Additional Drivers and select the displayed NVIDIA driver. 
- Reboot. 
- This is enough to setup Nvidia driver that your system supports. Check the installation by running nvidia-smi in the terminal.

## Step 3: Install development and runtime libraries
```
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.2.24-1+cuda10.0  \
    libcudnn7-dev=7.6.2.24-1+cuda10.0
```
The above step installs the development and runtime libraries. Kindly, replace with the respective version of cuda and libcudnn, if you have installed any other version of CUDA in the above steps. Also, make sure to install the compatible libcudnn version for the installed CUDA.

## Step 4: Install Tensorflow and check!
```
pip install tensorflow-gpu
```
The above installation should complete without any errors and the import statement should work properly after installation. 
That's it...we completed the baby step for our journey towards machine learning and deep learning. 

## Things to Keep in mind

- Make sure that you are installing the compatible versions of CUDA and Ubuntu vesrions. There will be cases where some version of CUDA might have not been released for Ubuntu 18.04 or 16.04
- Feel free to install all these system building libraries for the installation to proceed smoothly.
```
sudo apt-get update 

sudo apt-get -y upgrade 

sudo apt-get install -y build-essential cmake git unzip pkg-config 

sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev 

sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev 

sudo apt-get install -y libxvidcore-dev libx264-dev 

sudo apt-get install -y libgtk-3-dev 

sudo apt-get install -y libhdf5-serial-dev graphviz 

sudo apt-get install -y libopenblas-dev libatlas-base-dev gfortran 

sudo apt-get install -y python-tk python3-tk python-imaging-tk 

sudo apt-get install -y python2.7-dev python3-dev 

sudo apt-get install -y linux-image-generic linux-image-extra-virtual sudo apt-get install -y linux-source linux-headers-generic
```

## References

- https://kernel.ubuntu.com/~kernel-ppa/mainline/v4.15/

- https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/

