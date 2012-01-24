#!/usr/bin/env bash

${CMAKE_INSTALL_PREFIX}/bin/physisc --mpi-opencl \
    -I${CMAKE_INSTALL_PREFIX}/include \
    $*
