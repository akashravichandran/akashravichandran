---
title: Installing any version of CUDA the easy way!
date: 2020-05-22 10:00:00 +07:00
tags: [python, conda, installation]
description: Installing CUDA the easy way fot getting started with ML/DL.
categories: getting-started
---

# Why am I writing this?
Installing CUDA is not an easy task for various reasons becasue of its issues with respect to version incompability, driver issues and many more. After going through those definite steps to get your CUDA working, there is another simple way to do the same. I will be explaining you how to do this in this blog post.

# Why Project Environments?
It is highly unlikely for anyone to use the different versions of the same library in a project. Maintaining a virtaul environment for all your project is a good practise. Virtual Environments gives us the opportunity to package our libraries, so that it is accessible for any other project which requires the same set of libraries. The important thing to note here is that our virtual environments dont disturb our base environments.

# About Conda
Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies. Conda easily creates, saves, loads and switches between environments on your local computer. It was created for Python programs, but it can package and distribute software for any language.

Conda as a package manager helps you find and install packages. If you need a package that requires a different version of Python, you do not need to switch to a different environment manager, because conda is also an environment manager. With just a few commands, you can set up a totally separate environment to run that different version of Python, while continuing to run your usual version of Python in your normal environment.

# Installing the latest driver for your Ubuntu Desktop
**Ubuntu Install Nvidia driver using the CLI method**

```
- apt search nvidia-driver
- sudo apt install nvidia-driver
- sudo reboot
- **verification** - nvidia-smi
```

# Creating the Virtual Environment using CUDA!

**CUDA 9.2**

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch

**CUDA 10.0**

conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

**CUDA 10.1**

conda create --name tf_gpu cudatoolkit=10.1 tensorflow-gpu

**CPU Only**

conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch

The above commands help in creating our virtualenv by downloading the required dependancies and the specific cudatoolkit version that we might need and also installing the library that we need with GPU support.

# Conclusion
Thus, we have used a simple one line command to setup our environment for datascience projects. Hope you liked this! Thank you.

# References
- [ubuntu-linux-install-nvidia-driver-latest-proprietary-driver](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)
- [conda-cheatsheet.pdf](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
- [tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)