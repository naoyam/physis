#!/usr/bin/env bash

${CMAKE_INSTALL_PREFIX}/bin/physisc --cuda-host-memory -I${CMAKE_INSTALL_PREFIX}/include $*
