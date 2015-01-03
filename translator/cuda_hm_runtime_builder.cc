// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/cuda_hm_runtime_builder.h"

namespace physis {
namespace translator {

CUDAHMRuntimeBuilder::CUDAHMRuntimeBuilder(SgScopeStatement *global_scope,
                                           const Configuration &config):
    CUDARuntimeBuilder(global_scope, config) {
}

} // namespace translator
} // namespace physis


