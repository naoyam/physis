// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_TRANSLATOR_H_

#include "translator/translator.h"
#include "translator/translator_common.h"
#include "translator/mpi_translator.h"

namespace physis {
namespace translator {

class MPICUDAOptimizer;

class MPICUDATranslator: public MPITranslator {
 public:
  MPICUDATranslator(const Configuration &config);
  virtual ~MPICUDATranslator() {}
  virtual SgBasicBlock *generateRunKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg);
  virtual SgBasicBlock *generateRunBoundaryKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg);
  virtual SgBasicBlock *generateRunMultiStreamBoundaryKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg, int dim, bool fw);
  virtual SgBasicBlock *generateRunInnerKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg);
  friend class MPICUDAOptimizer;  
 protected:
  int block_dim_x_;
  int block_dim_y_;
  int block_dim_z_;
  bool flag_using_dimy_as_dimz_;
  bool flag_pre_calc_grid_address_;
  bool flag_multistream_boundary_;
  string boundary_kernel_width_name_;
  string inner_prefix_;
  string boundary_suffix_;
  // REFACTORING: same as CUDATranslator
  virtual SgExpression *BuildBlockDimX();
  virtual SgExpression *BuildBlockDimY();
  virtual SgExpression *BuildBlockDimZ();
  virtual SgIfStmt *BuildDomainInclusionCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgExpression *dom_ref) const;
  virtual SgIfStmt *BuildDomainInclusionInnerCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgExpression *dom_ref, SgExpression *width,
      SgStatement *ifclause) const;
  virtual void ProcessStencilMap(StencilMap *smap, SgVarRefExp *stencils,
                                 int stencil_index, Run *run,
                                 SgScopeStatement *function_body,
                                 SgScopeStatement *loop_body,
                                 SgVariableDeclaration *block_dim);
  virtual SgBasicBlock *generateRunBody(Run *run); 

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
  virtual SgType *BuildOnDeviceGridType(GridType *gt);
  virtual void translateKernelDeclaration(SgFunctionDeclaration *node);
  virtual SgFunctionDeclaration *generateRunKernel(StencilMap *s);
  virtual std::vector<SgFunctionDeclaration*> generateRunBoundaryKernel(StencilMap *s);
  virtual std::vector<SgFunctionDeclaration*> generateRunMultiStreamBoundaryKernel(
      StencilMap *s);
  virtual SgFunctionDeclaration *generateRunInnerKernel(StencilMap *s);
  virtual SgFunctionCallExp* generateKernelCall(
      StencilMap *stencil, SgExpressionPtrList &indexArgs);
  virtual SgExprListExp* generateKernelCallArgList(
      StencilMap *stencil, SgExpressionPtrList &indexArgs);
  virtual void GenerateInnerKernel(SgFunctionDeclaration *original);
  virtual void GenerateBoundaryKernel(SgFunctionDeclaration *original);  
  std::string GetBoundarySuffix(int dim, bool fw);
  std::string GetBoundarySuffix();
  virtual bool translateGetKernel(SgFunctionCallExp *node,
                                  SgInitializedName *gv);
  std::set<SgFunctionSymbol*> cache_config_done_;
  void BuildFunctionParamList(SgClassDefinition *param_struct_def,
                              SgFunctionParameterList *&params,
                              SgInitializedName *&grid_arg,
                              SgInitializedName *&dom_arg);

};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_TRANSLATOR_H_ */
