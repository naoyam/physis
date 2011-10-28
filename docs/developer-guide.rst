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
<BUILD_DIRECTORY>/testing/run_tests.sh. For example, to run all tests
for all targets, use the driver as::

  ./testing/run_tests

To run just a specific test::

  ./testing/run_tests -s <test-name>

where <test-name> is the base name of the test file name (e.g.,
test_1). To run all tests only against some specific target::

  ./testing/run_tests -t <target>

The driver also allows for command-line configurations of MPI process
number and domain decomposition of each test program. For
example, to use 16 processes with the decomposion of 16, 4x4, 4x2x2,
for 1-D, 2-D, and 3-D problems, respectively, use the proc-dim option
like::

  ./testing/run_tests --proc-dim 16,4x4,4x2x2

Multiple values are allowed for many options of the test driver. For
example, these are all valid options::

  ./testing/run_tests --proc-dim '16,4x4,4x2x2 2,1x2,2x1x1'
  ./testing/run_tests -t 'cuda mpi'
  ./testing/run_tests -s 'test_1 test_2' -t 'ref mpi'

For more information on the usage of the driver, see help by::

  ./testing/run_tests -h

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