# Building Physis

## Prerequisites
* Cmake
* Rose
    * Rose requires Boost, JDK, and several other packages. See docs/rose.md.
* MPI
    * OpenMPI
        * --disable-visibility option needs to be passed to the configure script
	* Or define OMPI_C_HAVE_VISIBILITY 0 before including mpi.h
	* See https://mailman.nersc.gov/pipermail/rose-public/2010-July/000314.html  
* Lua
    * MacOS
        * brew instal lua
    * Ubuntu
        * apt-get install liblua5.1-dev
* CUDA
    * Both toolkit and SDK (tested with 3.2)
    * Specify the location of SDK by environment variable NVSDK_ROOT  
    * MacOSX's SDK may not contain 64-bit version of cutil. It can be
      build by entering C/lib/common directory and type make x86_64=1
    
## Steps
1. Change directory to an empty build directory
2. Run 'cmake -i <path-to-src>', where <path-to-src> is the path to
  directory rose under the unpacked source. Specifing locations of
  other dependencies may be needed.
    * Example on Ubuntu
        * NVSDKCOMPUTE_ROOT=/home/naoya/projects/cuda/sdk3.2/C
	cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/../install
        -DCMAKE_PREFIX_PATH=$HOME/projects/tools/rose/git ../src
    * On Tsubame
        * (CMake variable) CMAKE_PREFIX_PATH=/work0/GSIC/apps/boost/1_45_0/gcc
        * (CMake variable) JAVA_JVM_LIBRARY_DIRECTORIES=/usr/lib64/jvm/java/jre/bin/classic
        * (CMake variable) JAVA_INCLUDE_PATH2=/usr/lib64/jvm/java/include
        * (CMake variable) CMAKE_PREFIX_PATH=$HOME/projects/tools/rose/install
        * (shell variable) NVSDKCOMPUTE_ROOT=/home/naoya/projects/cuda/sdk3.2
3. Cmake then will search for the location of Boost, Java, and Rose. If
  Boost and JDK are already installed, they should be detected by
  Cmake. The path to a Rose installation must be supplied
  manually into a prompt issued by Cmake. Other path options such as
  install path can be left as is at this time.
4. Run make. This should produce executable 'physis' under the build
   directory. 

## Hints
* Once cmake is run and build files are generated, the make command is the
  only command needed to be run. Cmake is automatically kicked by the
  Makefiles if necessary.
* Building agaist ROSE may take significant time. Consider using
  ccache for reducing compilation time. 
  - Install ccache
  - cmake -D CMAKE_CXX_COMPILER=ccache  -D CMAKE_CXX_COMPILER_ARG1=c++  -i .. 

