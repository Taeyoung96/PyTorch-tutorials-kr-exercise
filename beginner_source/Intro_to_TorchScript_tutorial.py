"""
Introduction to TorchScript
===========================

*James Reed (jamesreed@fb.com), Michael Suo (suo@fb.com)*, rev2  

**번역** : `김태영<https://github.com/Taeyoung96>`

이 튜토리얼은 C++와 같은 고성능 환경에서 실행할 수 있는   
Pytorch 모델(``nn.Module``의 하위클래스)의 중간 표현 방식인 TorchScript에 대한 소개입니다.


이 튜토리얼에서 우리는 다음과 같은 내용을 다룹니다:

1. Pytorch에서 모델을 만들 때 기초가 되는 것들:  

-  모듈들 (Modules)
-  ``forward``함수를 정의  
-  모듈을 계층 구조로 구성

2. Pytorch 모듈들을 우리의 고성능 배포 런타임인 TorchSript로 변환하는 구체적인 방법들

-  기존 모듈 트레이싱(Tracing)하기
-  스크립팅(Scripting)을 활용하여 모듈 직접 컴파일 하기  
-  두 접근 방식을 모두 구성하는 방법  
-  TorchScript 모듈들을 활용하여 저장하고 불러오기

이 튜토리얼을 완료한 후에는 실제로 C++에서 TorchScript 모델을 불러오는 예제를 안내하는  
`후속 튜토리얼<https://pytorch.org/tutorials/advanced/cpp_export.html>`_을 진행할 수 있습니다.

"""

import torch  # 이것은 PyTorch와 TorchSript를 사용할 때 꼭 필요로 합니다!
print(torch.__version__)


######################################################################
# PyTorch 모델을 만들 때 기초가 되는 것들
# ---------------------------------
#
# 간단한 ``Module``을 정의하는 것부터 시작하겠습니다.  
# ``Module``은 PyTorch의 기본 구성 단위입니다.  
# 다음과 같은 내용을 포함하고 있습니다.
#
# 1. 호출이 일어날 때 모듈을 준비하는 생성자
# 2. ``Parameters``와 하위 ``Modules``의 집합체.  
#    이것들은 생성자를 통해 초기화되고 호출 중에 모듈에 의해 사용될 수 있습니다.
# 3. ``forward`` 함수. 이것은 모듈이 호출될 때 실행되는 코드입니다.
#
# 작은 예제로 시작해보겠습니다:
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))


######################################################################
# 따라서 우리는 다음 작업을 수행했습니다:
#
# 1. ``torch.nn.Module``를 하위 클래스로 갖는 클래스를 만들었습니다.  
# 2. 생성자를 정의 했습니다. 생성자는 많은 일을 하지 않고, ``super``로 생성자를 호출합니다.  
# 3. 2개의 입력을 받아 2개의 출력을 반환하는 ``forward`` 함수를 정의해봅시다.  
#    ``forward`` 함수의 실제 내용은 사실상 중요하지 않습니다만,  
#    이것은 가짜 `RNN cell <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`입니다.  
#    즉, 반복(Loop)에 적용되는 함수입니다.  
#
# 우리는 모듈을 생성하고, 3x4 크기의 무작위 값들로 이루어진 행렬 ``x``와 ``h``를 만들었습니다.  
# 그리고 우리는 ``my_cell(x, h)`` 를 이용해 cell을 호출했습니다.  
# 이것은 우리의 ``forward`` 함수를 차례로 호출합니다.
#
# 좀 더 흥미로운 일을 진행해봅시다:
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))


######################################################################
# We’ve redefined our module ``MyCell``, but this time we’ve added a
# ``self.linear`` attribute, and we invoke ``self.linear`` in the forward
# function.
#
# What exactly is happening here? ``torch.nn.Linear`` is a ``Module`` from
# the PyTorch standard library. Just like ``MyCell``, it can be invoked
# using the call syntax. We are building a hierarchy of ``Module``\ s.
#
# ``print`` on a ``Module`` will give a visual representation of the
# ``Module``\ ’s subclass hierarchy. In our example, we can see our
# ``Linear`` subclass and its parameters.
#
# By composing ``Module``\ s in this way, we can succintly and readably
# author models with reusable components.
#
# You may have noticed ``grad_fn`` on the outputs. This is a detail of
# PyTorch’s method of automatic differentiation, called
# `autograd <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`__.
# In short, this system allows us to compute derivatives through
# potentially complex programs. The design allows for a massive amount of
# flexibility in model authoring.
#
# Now let’s examine said flexibility:
#

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))


######################################################################
# We’ve once again redefined our MyCell class, but here we’ve defined
# ``MyDecisionGate``. This module utilizes **control flow**. Control flow
# consists of things like loops and ``if``-statements.
#
# Many frameworks take the approach of computing symbolic derivatives
# given a full program representation. However, in PyTorch, we use a
# gradient tape. We record operations as they occur, and replay them
# backwards in computing derivatives. In this way, the framework does not
# have to explicitly define derivatives for all constructs in the
# language.
#
# .. figure:: https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/dynamic_graph.gif
#    :alt: How autograd works
#
#    How autograd works
#


######################################################################
# Basics of TorchScript
# ---------------------
#
# Now let’s take our running example and see how we can apply TorchScript.
#
# In short, TorchScript provides tools to capture the definition of your
# model, even in light of the flexible and dynamic nature of PyTorch.
# Let’s begin by examining what we call **tracing**.
#
# Tracing ``Modules``
# ~~~~~~~~~~~~~~~~~~~
#

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)


######################################################################
# We’ve rewinded a bit and taken the second version of our ``MyCell``
# class. As before, we’ve instantiated it, but this time, we’ve called
# ``torch.jit.trace``, passed in the ``Module``, and passed in *example
# inputs* the network might see.
#
# What exactly has this done? It has invoked the ``Module``, recorded the
# operations that occured when the ``Module`` was run, and created an
# instance of ``torch.jit.ScriptModule`` (of which ``TracedModule`` is an
# instance)
#
# TorchScript records its definitions in an Intermediate Representation
# (or IR), commonly referred to in Deep learning as a *graph*. We can
# examine the graph with the ``.graph`` property:
#

print(traced_cell.graph)


######################################################################
# However, this is a very low-level representation and most of the
# information contained in the graph is not useful for end users. Instead,
# we can use the ``.code`` property to give a Python-syntax interpretation
# of the code:
#

print(traced_cell.code)


######################################################################
# So **why** did we do all this? There are several reasons:
#
# 1. TorchScript code can be invoked in its own interpreter, which is
#    basically a restricted Python interpreter. This interpreter does not
#    acquire the Global Interpreter Lock, and so many requests can be
#    processed on the same instance simultaneously.
# 2. This format allows us to save the whole model to disk and load it
#    into another environment, such as in a server written in a language
#    other than Python
# 3. TorchScript gives us a representation in which we can do compiler
#    optimizations on the code to provide more efficient execution
# 4. TorchScript allows us to interface with many backend/device runtimes
#    that require a broader view of the program than individual operators.
#
# We can see that invoking ``traced_cell`` produces the same results as
# the Python module:
#

print(my_cell(x, h))
print(traced_cell(x, h))


######################################################################
# Using Scripting to Convert Modules
# ----------------------------------
#
# There’s a reason we used version two of our module, and not the one with
# the control-flow-laden submodule. Let’s examine that now:
#

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)


######################################################################
# Looking at the ``.code`` output, we can see that the ``if-else`` branch
# is nowhere to be found! Why? Tracing does exactly what we said it would:
# run the code, record the operations *that happen* and construct a
# ScriptModule that does exactly that. Unfortunately, things like control
# flow are erased.
#
# How can we faithfully represent this module in TorchScript? We provide a
# **script compiler**, which does direct analysis of your Python source
# code to transform it into TorchScript. Let’s convert ``MyDecisionGate``
# using the script compiler:
#

scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)


######################################################################
# Hooray! We’ve now faithfully captured the behavior of our program in
# TorchScript. Let’s now try running the program:
#

# New inputs
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell(x, h)


######################################################################
# Mixing Scripting and Tracing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Some situations call for using tracing rather than scripting (e.g. a
# module has many architectural decisions that are made based on constant
# Python values that we would like to not appear in TorchScript). In this
# case, scripting can be composed with tracing: ``torch.jit.script`` will
# inline the code for a traced module, and tracing will inline the code
# for a scripted module.
#
# An example of the first case:
#

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)



######################################################################
# And an example of the second case:
#

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)


######################################################################
# This way, scripting and tracing can be used when the situation calls for
# each of them and used together.
#
# Saving and Loading models
# -------------------------
#
# We provide APIs to save and load TorchScript modules to/from disk in an
# archive format. This format includes code, parameters, attributes, and
# debug information, meaning that the archive is a freestanding
# representation of the model that can be loaded in an entirely separate
# process. Let’s save and load our wrapped RNN module:
#

traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)


######################################################################
# As you can see, serialization preserves the module hierarchy and the
# code we’ve been examining throughout. The model can also be loaded, for
# example, `into
# C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`__ for
# python-free execution.
#
# Further Reading
# ~~~~~~~~~~~~~~~
#
# We’ve completed our tutorial! For a more involved demonstration, check
# out the NeurIPS demo for converting machine translation models using
# TorchScript:
# https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ
#
