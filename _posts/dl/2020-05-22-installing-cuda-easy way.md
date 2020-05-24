---
title: Installing CUDA the easy way!
date: 2020-05-22 10:00:00 +07:00
tags: [python, conda, installation]
description: Installing CUDA the easy way fot getting started with ML/DL.
categories: Getting Started
---

# Why am I writing this?
Installing CUDA is not an easy task for various reasons becasue of its issues with respect to version incompability, driver issues and many more. After going through those definite steps to get your CUDA working, there is another simple way to do the same. I will be explaining you how to do this in this blog post.

# Why Project Environments?
It is highly unlikely for anyone to use the different versions of the same library in a project. Maintaining a virtaul environment for all your project is a must. Virtual Environments gives us the opportunity to package our libraries, so that it is accessible for any other project which requires the same set of libraries. The important thing to note here is that our virtual environments dont disturb our base environments.

# Creating the Virtual Environment using CUDA!

conda create --name tf_gpu tensorflow-gpu

The above command helps in creating our virtualenv which helps in downloading the required dependancies for the GPU as well installing the library that we need with GPU support.

# References

https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc