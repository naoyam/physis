// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_MPI_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_MPI_TRANSLATOR_H_

#include "translator/translator.h"
#include "translator/translator_common.h"
#include "translator/reference_translator.h"
#include "translator/mpi_runtime_builder.h"

namespace physis {
namespace translator {

class MPITranslator: public ReferenceTranslator {
 public:
  MPITranslator(const Configuration &config);
  virtual ~MPITranslator() {}
  virtual void run();
 protected:
  MPIRuntimeBuilder *mpi_rt_builder_;
  bool flag_mpi_overlap_;
  virtual void translateInit(SgFunctionCallExp *node);
  virtual void translateRun(SgFunctionCallExp *node,
                            Run *run);
  virtual SgBasicBlock *generateRunBody(Run *run);
  virtual SgFunctionDeclaration *generateRun(Run *run);
  virtual void appendNewArgExtra(SgExprListExp *args, Grid *g);
  virtual bool translateGetKernel(SgFunctionCallExp *node,
                                  SgInitializedName *gv);
  virtual bool translateGetHost(SgFunctionCallExp *node,
                                SgInitializedName *gv);
  virtual void translateEmit(SgFunctionCallExp *node, SgInitializedName *gv);
  virtual void GenerateLoadRemoteGridRegion(
      StencilMap *smap, SgExpression *stencil_ref,
      Run *run, SgScopeStatement *scope,
      SgInitializedNamePtrList &remote_grids,
      SgStatementPtrList &statements,
      bool &overlap_eligible,
      int &overlap_width);
  virtual void ProcessStencilMap(StencilMap *smap, SgVarRefExp *stencils,
                                 int stencil_index, Run *run,
                                 SgScopeStatement *function_body,
                                 SgScopeStatement *loop_body);
  virtual void DeactivateRemoteGrids(
      StencilMap *smap,
      SgExpression *stencil_ref,
      SgScopeStatement *scope,
      const SgInitializedNamePtrList &remote_grids);
  virtual void FixGridAddresses(StencilMap *smap,
                                SgExpression *stencil_ref,
                                SgScopeStatement *scope);
  virtual void CheckSizes();

  int global_num_dims_;
  //IntArray global_size_;
  SgFunctionSymbol *stencil_run_func_;
  string get_addr_name_;
  string get_addr_no_halo_name_;
  string emit_addr_name_;
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_MPI_TRANSLATOR_H_ */
