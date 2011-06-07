// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_

#include "translator/reference_translator.h"

namespace physis {
namespace translator {

class CUDATranslator : public ReferenceTranslator {
 public:
  CUDATranslator(const Configuration &config);
  virtual ~CUDATranslator() {}
  virtual void run();
  
 protected:  
  int block_dim_x_;
  int block_dim_y_;
  int block_dim_z_;

  // If flag_using_dimy_as_dimz_ is 'false', the Translator generates
  // a simple z-loop, which covers entiry z dimention, for kernel invocation
  // of 3d-grid.
  // i.e.
  //   grid_block = dim3(nx/dim_x, ny/dim_y, 1);
  //   x = blockIdx.x * blockDim.x + threadIdx.x;
  //   y = blockIdx.y * blockDim.y + threadIdx.y;
  //   for (z = 0; z < nz; z++) {
  //     kernel(x, y, z, ...);
  //   }
  // On the otherhand, if the flag is 'true', it uses a trick to reduce
  // size of each thread by dividing z-loop into smaller loops.
  // i.e.
  //   grid_block = dim3(nx*ny/(dim_x*dim_y), nz/dim_z, 1);
  //   blockIdx_y = blockIdx.x / (nx/dim_x);
  //   blockIdx_x = blockIdx.x - blockIdx_y*(nx/dim_x);
  //   x = dim_x * blockIdx_x + threadIdx.x;
  //   y = dim_y * blockIdx_y + threadIdx.y;
  //   z = dim_z * blockIdx.y;
  //   for (k = 0; k < dim_z; k++) {
  //     kernel(x, y, z+k, ...);
  //   }
  bool flag_using_dimy_as_dimz_;

  // If this flag is 'true', the Translator generates for grid_get in the
  // begining of the kernel. This optimization is effective for some cases such
  // as consecutive code block of 'if' and 'grid_get'. In which case, size of
  // if block could be reduced.
  // i.e.
  //   kernel(x, y, z, g, ...) {
  //     ...
  //     val = grid_get(g, x, y, z);
  //     ...
  //   }
  //   It generates code as following
  //   kernel(x, y, z, g, ...) {
  //     float *p = (float *)(g->p0) + z*nx*ny + y*nx + x;
  //     ...
  //     val = *p;
  //     ...
  //   }
  bool flag_pre_calc_grid_address_;

  virtual SgVariableDeclaration *generateGridDimDeclaration2D(
      const SgName &name,
      SgExpression *stencil_var,
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL);

  virtual SgVariableDeclaration *generateGridDimDeclaration3D(
      const SgName &name,
      SgExpression *stencil_var,
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL);

  virtual void translateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void translateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool isKernel);
  //virtual void translateSet(SgFunctionCallExp *node, SgInitializedName *gv);  
  virtual SgBasicBlock *generateRunBody(Run *run);
  virtual SgBasicBlock *GenerateRunLoopBody(Run *run,
                                            SgScopeStatement *outer_block);  
  virtual SgFunctionDeclaration *generateRunKernel(StencilMap *s);
  virtual SgBasicBlock *generateRunKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg);
  virtual SgFunctionCallExp* generateKernelCall(
      StencilMap *stencil, SgExpressionPtrList &indexArgs,
      SgScopeStatement *containingScope);
  virtual SgExpression *BuildBlockDimX();
  virtual SgExpression *BuildBlockDimY();
  virtual SgExpression *BuildBlockDimZ();
  virtual SgIfStmt *BuildDomainInclusionCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgExpression *dom_ref) const;
  virtual SgType *BuildOnDeviceGridType(GridType *gt);
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_ */
