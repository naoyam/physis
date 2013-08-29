// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "translator/cuda_hm_runtime_builder.h"

namespace physis {
namespace translator {

CUDAHMRuntimeBuilder::CUDAHMRuntimeBuilder(SgScopeStatement *global_scope):
    CUDARuntimeBuilder(global_scope) {
}

} // namespace translator
} // namespace physis


