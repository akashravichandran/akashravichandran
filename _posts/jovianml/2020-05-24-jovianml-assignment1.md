---
title: JovianML - ZerotoGAN - Assignment 1
date: 2020-05-24
tags: [python, conda, jupyter, google colab, pytorch]
description: My solutions for assignment one of jovianml zerotogan
categories: jovianml
---


<a href="https://colab.research.google.com/github/akashravichandran/jovian-zerotogan/blob/master/01_tensor_operations.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


### Some useful methods from Pytorch tensor class

Here I will be discussing some of the useful pytorch methods that will be useful for your datascience projects.
- torch.add
- torch.cat
- apply_
- torch.abs
- torch.arange


```python
# Import torch and other required modules
import torch
```

**torch.tensor() always copies data. If you have a Tensor data and just want to change its requires_grad flag, use requires_grad_() or detach() to avoid a copy. If you have a numpy array and want to avoid a copy, use torch.as_tensor().**



## Function 1 - torch.add

Add a scalar or tensor to self tensor. If both alpha and other are specified, each element of other is scaled by alpha before being used.

When other is a tensor, the shape of other must be broadcastable with the shape of the underlying tensor




```python
# Example 1 - working
a = torch.ones(1)
b = torch.ones(1)
torch.add(a, b)
```




    tensor([2.])



The first example adds a one dimensional tensor and returns the answer as a one dimensional tensor


```python
# Example 2 - working
a = torch.randn(4)
b = torch.randn(4)
torch.add(a, b)
```




    tensor([-0.0584,  1.8820, -0.0978,  1.1470])



The second example adds a one dimensional four elements to another one dimensional four elements tensor and returns the answer as a four element tensor.

```python
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working (change this)
a = torch.randn(4)
b = torch.randn(2)
torch.add(a, b)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-23-7ed36789bd3e> in <module>()
          3 a = torch.randn(4)
          4 b = torch.randn(2)
    ----> 5 torch.add(a, b)
    

    RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 0


The third example when the dimension of the operating tensor vary, they throw a runtime error

## Function 2 - torch.cat

Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.


```python
# Example 1 - working
x = torch.randn(1, 2)
torch.cat((x, x), 0)
```




    tensor([[-0.2554, -0.1760],
            [-0.2554, -0.1760]])



The first example concatenates the elements of the tensor by appending to the vertical dimension


```python
# Example 2 - working
x = torch.randn(1, 2)
torch.cat((x, x), 1)
```




    tensor([[-0.8917,  0.3384, -0.8917,  0.3384]])



The second example concatenates the elements of the tensor by appending to the horizontal dimension


```python
# Example 3 - breaking (to illustrate when it breaks)
x = torch.randn(1, 2)
y = torch.randn(2, 2)
torch.cat((x, y), 1)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-14-b6855dba74e8> in <module>()
          2 x = torch.randn(1, 2)
          3 y = torch.randn(2, 2)
    ----> 4 torch.cat((x, y), 1)
    

    RuntimeError: Sizes of tensors must match except in dimension 1. Got 1 and 2 in dimension 0


The third example breaks when applied to tensors of varying sizes.





```
# This is formatted as code
```

## Function 3 - apply_(callable) → Tensor

Applies the function callable to each element in the tensor, replacing each element with the value returned by callable.


```python
# Example 1 - working
def square(x):
  return x * x
a = 2 * torch.ones(1, 2)
a.apply_(square)
```




    tensor([[4., 4.]])



The first example applies the square function to the elements of the tensor


```python
# Example 2 - working
def cube(x):
  return x * x * x
a = 2 * torch.ones(1, 2)
a.apply_(cube)
```




    tensor([[8., 8.]])



The second example applies the cube function to the tensor at hand


```python
# Example 3 - breaking (to illustrate when it breaks)
def square(x):
  return x * x
cuda0 = torch.device('cuda:0')
a = torch.ones([1, 2], dtype=torch.float64, device=cuda0)
a.apply_(square)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-12-d67ec982cf04> in <module>()
          4 cuda0 = torch.device('cuda:0')
          5 a = torch.ones([2, 4], dtype=torch.float64, device=cuda0)
    ----> 6 a.apply_(square)
    

    TypeError: apply_ is only implemented on CPU tensors


The third example breaks when applied to the tensor of devicetype gpu, because of its incompatibility.


## Function 4 - torch.abs

Computes the element-wise absolute value of the given input tensor.


```python
# Example 1 - working
torch.abs(torch.tensor(-3))

```




    tensor(3)



The first example applies the abs function to a scalar tensor


```python
# Example 2 - working
torch.abs(torch.tensor([-1, -2, 3]))
```




    tensor([1, 2, 3])



The second example applies the abs function to a one dimensional tensor


```python
# Example 3 - breaking (to illustrate when it breaks)
torch.abs(torch.tensor([True, True, False]))
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-34-84266fd23906> in <module>()
          1 # Example 3 - breaking (to illustrate when it breaks)
    ----> 2 torch.abs(torch.tensor([True, True, False]))
    

    RuntimeError: "abs_cpu" not implemented for 'Bool'


The third example breaks when abs is applied to boolean.


## Function 5 - torch.arange





```python
# Example 1 - working
torch.arange(1, 2.5)
```




    tensor([1., 2.])



The first example gets a range of values without step specified


```python
# Example 2 - working
torch.arange(1, 2.5, 0.5)
```




    tensor([1.0000, 1.5000, 2.0000])



The second example gets a range of values with step specified


```python
# Example 3 - breaking (to illustrate when it breaks)
torch.arange(1, 2.5, 0)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-20-4fccf7bee718> in <module>()
          1 # Example 3 - breaking (to illustrate when it breaks)
          2 
    ----> 3 torch.arange(1, 2.5, 0)
    

    RuntimeError: step must be nonzero


The third example breaks when step value is set to zero


## Conclusion

Here I have randomly picked some methods from pytorch, pytorch is really vast and it is still growing. Hope you like the methods that I have taken here to be explained. Thank you!

## Reference Links
* Documentation for `torch.Tensor`: [https://pytorch.org/docs/stable/tensors.html](https://pytorch.org/docs/stable/tensors.html)

