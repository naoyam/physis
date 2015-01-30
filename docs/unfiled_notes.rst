* Boost v1.45 (or older) will cause "dereferencing type-punned
  pointer" warning with runtime/reduce.h when gcc optimization is
  enabled. It was fixed as discussed here:
  https://svn.boost.org/trac/boost/ticket/4538. Version 1.47 seems to 
  have that fix.
* On Mac OS X, translator is not tested since building ROSE on Mac OS X
  is not well supported.
* On Mac OS X, the Boost library installed by Homebrew may be built
  with g++, not the OS X default clang++. nvcc works only with the
  default compiler, so the Homebrew-built Boost and nvcc do not work
  together.
* On Mac OS X, nvcc (at least v6.5) uses stdlibc++, not the default
  libc++. Linking by CMake uses c++ rather than nvcc, so -stdlib flag
  needs to be set.
* MPICH fails to compile with OS X c++ with -stdlib=libstdc++ switch. 
