========================
Physis Developer's Guide
========================

This document is intended to provide information for developers of the
Physis framework. To find how to use the framework, see programming.md
and compilation.md.

This document is work in progress, and is currently just a collection
of useful tips and notes.

Testing
-------

The testing/tests directory contains several Physis programs that
are written to verify certain parts of the framework. To run the
tests, use the test driver at
<BUILD_DIRECTORY>/run_system_tests.sh. For example, to run all tests
for all targets, use the driver as::

  ./run_system_tests

To run just a specific test::

  ./run_system_tests -s <test-name>

where <test-name> is the base name of the test file name (e.g.,
test_1). To run all tests only against some specific target::

  ./run_system_tests -t <target>

The driver also allows for command-line configurations of MPI process
number and domain decomposition of each test program. For
example, to use 16 processes with the decomposion of 16, 4x4, 4x2x2,
for 1-D, 2-D, and 3-D problems, respectively, use the proc-dim option
like::

  ./run_system_tests --proc-dim 16,4x4,4x2x2

Multiple values are allowed for many options of the test driver. For
example, these are all valid options::

  ./run_system_tests --proc-dim '16,4x4,4x2x2 2,1x2,2x1x1'
  ./run_system_tests -t 'cuda mpi'
  ./run_system_tests -s 'test_1 test_2' -t 'ref mpi'

For more information on the usage of the driver, see help by::

  ./run_system_tests -h

Each combination of the specified cases is translated, compiled, and
executed by the test driver. This process may be done multiple times
if the translator for a target has configuration options (e.g.,
MPI_OVERLAP). The driver generates a fixed set of option patterns
dynamically, and run the above process for each option pattern. To
modify the tested option patterns, the driver script needs to be
modified. 

When the driver is executed, all intermediate files are generated
under a directory named "test_output/TIMESTAMP". A log file is also
created at the current working directory.

Coding Style
------------

For C/C++ code in Physis, we adopt the Google C++ Style Guide
(http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml). We
initially didn't have any specific coding rules, and so part of the
code still don't follow the the guide. We will gradually clean up
non-conforming code.

Git Repository and Branching Model
----------------------------------

The repository is hosted at http://github.com/naoyam/physis. You can
browse the code and its repository history there.

For the development of Physis, we adopt the Git branching model by
Vincent Driessen. The main development work is on the `develop`
branch, and when each new feature is added, a new branch named
"feature/feature-name" is created. The development of the feature is
done on the feature branch, which is then merged back to the `develop`
branch when finished. See
http://nvie.com/posts/a-successful-git-branching-model/ for more
information.

Data Types
----------

Some common data types are defined in the physis_common.h file. 

- PSIndex: Integral data type for grid indices (int32_t or int64_t)
- PSVectorInt: Array of int with PS_MAX_DIM elements

In addition, internal_common.h defines several types for Physis
translators and runtimes only.

- IntArray: boost::array of int
- UnsignedArray: boost::array of unsigned
- SizeArray: boost::array of size_t
- SSizeArray: boost::array of ssize_t
- IndexArray: boost::array of PSIndex
