#!/usr/bin/env bash

${CMAKE_INSTALL_PREFIX}/bin/physisc --opencl -DPHYSIS_OPENCL_HEADER_DIR=\"${CMAKE_INSTALL_PREFIX}/include\" -I${CMAKE_INSTALL_PREFIX}/include $@
