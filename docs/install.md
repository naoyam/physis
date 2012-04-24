# Building Physis

## Prerequisites
* Cmake 
    * version 2.8 or newer
* Boost
    * Boost is also used in ROSE, which only supports version 1.36 to 1.45.
* Lua
* GNU getopt	
* ROSE (Optional)
    * Required when building the translator. Only the runtime will be
      built if not found.
    * Rose requires Boost, JDK, and several other packages. See docs/rose.md.
* MPI (Optional)
    * Required for MPI-based runtimes. If not found, no MPI-based
      runtimes will be built.
* CUDA (Optional)
    * Required for CUDA-based runtimes. If not found, no CUDA-based
      runtimes will be built.
    * Both toolkit and SDK (tested with 4.0)
    * Specify the location of SDK by environment variable NVSDKCOMPUTE_ROOT  
    * MacOSX's SDK may not contain 64-bit version of cutil. It can be
      build by entering C/lib/common directory and type make x86_64=1
    
## Steps
1. Change directory to an empty build directory
2. Set shell environment variable NVSDKCOMPUTE_ROOT as the path to the root SDK directory path.
3. Run cmkake as follows:
  'cmake -i PHYSIS_SOURCE_PATH -DCMAKE_INSTALL_PREFIX=PHYSIS_INSTALL_PATH -DCMAKE_PREFIX_PATH=ROSE_INSTALL_PATH -DBOOST_ROOT=BOOST_INSTALL_PATH' 
  where PHYSIS_SOURCE_PATH is the path to the Physis root directory, PHYSIS_INSTALL_PATH is the path where Physis should be installed, ROSE_INSTALL_PATH is the path where ROSE is installed. 
 On Tsubame, the following CMake variable might need to be defined.
    * JAVA_JVM_LIBRARY_DIRECTORIES=/usr/lib64/jvm/java/jre/bin/classic
    * JAVA_INCLUDE_PATH2=/usr/lib64/jvm/java/include
4. Cmake then will search for the location of Boost, Java, and Rose. If
  Boost and JDK are already installed, they should be detected by
  Cmake.
5. Run make. This should produce executable 'physis' under the build
   directory. 
6. Run make install.

## Hints
* Once cmake is run and build files are generated, the make command is the
  only command needed to be run. Cmake is automatically kicked by the
  Makefiles if necessary.
* Building agaist ROSE may take significant time. Consider using
  ccache for reducing compilation time. 
  - Install ccache
  - cmake -D CMAKE_CXX_COMPILER=ccache  -D CMAKE_CXX_COMPILER_ARG1=c++  -i .. 

