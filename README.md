# A High-Level Domain-Specific Framework for Stencil Computations

## Overview

Physis is a framework for stencil computations on modern heterogeneous
parallel computing systems. The main goals are productivity and
performance. The Physis DSL, which consists of a small set of custom
programming constructs , allows very concise and portable implementations of
stencil computations on a variety of high performance systems with a
particular focus on modern programmable GPUs. A single Physis program
runs on x86 CPUs, NVIDIA GPUs, and even clusters of them with no
platform-specific code.  

This software consists of a DSL translator and runtime for each target
platform. The translator automatically generates platform-specific
source code from a single Physis source, which is then compiled with a
platform-native compiler to generate final executable code. The
runtime component is a thin software layer that manages
application-independent tasks for each platform, such as managing GPU
devices and network connections. 

## Features

* C-based embedded DSL consisting of declarative intrinsics
* Platform portability
* Virtual shared memory
* No explicit parallel programming required
* Tested with the [fourth fastest supercomputer](http://tsubame.gsic.titech.ac.jp)

## Supported Platforms

* Linux and Mac OS X running on x86 CPUs
* CUDA-capable NVIDIA GPUs
* Clusters of machines with MPI

## Usage

### External Dependencies

Physis depends on the following external software:

* [GNU GCC C/C++ compiler](http://gcc.gnu.org/)
* [ROSE compiler infrastructure](http://www.rosecompiler.org/)
* [Boost C++ libraries](http://www.boost.org/)
* [CMake](http://www.cmake.org/)
* [Lua compiler and runtime libraries](http://www.lua.org)

In addition, the following platform-specific tools and libraries are
required when using the respective platform:

* [NVIDIA CUDA toolkit and SDK for using NVIDIA GPUs](http://developer.nvidia.com/cuda-downloads) (tested with version 3.2)
* MPI for multi-node parallel execution
  ([OpenMPI v1.4.2](http://www.open-mpi.org/) tested)

### Building Physis

See docs/build.md.
  
### Writing Physis Code

See docs/programming_physis.md. Some example code is available at the
examples directory.  

### Compiling Physis Programs

To be written.

## License (BSD License)

Copyright (c) 2011, Naoya Maruyama

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: 

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer. 
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the
  distribution. 
* Neither the name of the Tokyo Institute of Technology nor the names
  of its contributors may be used to endorse or promote products
  derived from this software without specific prior written
  permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
