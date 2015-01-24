// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/stencil_range.h"
#include "translator/reference_runtime_builder.h"
#include "translator/mpi_builder_interface.h"

namespace physis {
namespace translator {

class MPIRuntimeBuilder: virtual public ReferenceRuntimeBuilder,
                         virtual public MPIBuilderInterface {
 public:
  MPIRuntimeBuilder(SgScopeStatement *global_scope,
                    const Configuration &config,
                    BuilderInterface *delegator=NULL):
      ReferenceRuntimeBuilder(global_scope, config, delegator),
      flag_mpi_overlap_(false) {
    const pu::LuaValue *lv
        = config.Lookup(Configuration::MPI_OVERLAP);
    if (lv) {
      PSAssert(lv->get(flag_mpi_overlap_));
    }
    if (flag_mpi_overlap_) {
      LOG_INFO() << "Overlapping enabled\n";
    }
  }
  
  virtual ~MPIRuntimeBuilder() {}
  virtual SgFunctionCallExp *BuildIsRoot();
  virtual SgFunctionCallExp *BuildGetGridByID(SgExpression *id_exp);
  virtual SgFunctionCallExp *BuildDomainSetLocalSize(SgExpression *dom);

  virtual SgExpression *BuildGridBaseAddr(
      SgExpression *gvref, SgType *point_type);

  virtual SgFunctionParameterList *BuildRunFuncParameterList(Run *run);
  virtual void BuildRunFuncBody(
      Run *run, SgFunctionDeclaration *run_func);
  virtual void ProcessStencilMap(StencilMap *smap, SgVarRefExp *stencils,
                                 int stencil_index, Run *run,
                                 SgScopeStatement *function_body,
                                 SgScopeStatement *loop_body);
  
  // Derived from MPIBuilderInterface
  virtual void BuildLoadRemoteGridRegion(
      StencilMap *smap, SgVariableDeclaration *stencil_decl,
      Run *run, SgInitializedNamePtrList &remote_grids,
      SgStatementPtrList &statements, bool &overlap_eligible,
      int &overlap_width);
  // Derived from MPIBuilderInterface  
  virtual void BuildDeactivateRemoteGrids(
      StencilMap *smap,
      SgVariableDeclaration *stencil_decl,      
      const SgInitializedNamePtrList &remote_grids,
      SgStatementPtrList &stmt_list);
  
  virtual void BuildFixGridAddresses(StencilMap *smap,
                                     SgVariableDeclaration *stencil_decl,
                                     SgScopeStatement *scope);
  
  virtual bool IsOverlappingEnabled() const {
    return flag_mpi_overlap_;
  }

 protected:
  bool flag_mpi_overlap_;

  void BuildLoadRemoteGridRegion(
    SgInitializedName &grid_param,
    StencilRange &sr,
    StencilMap &smap,
    SgVariableDeclaration &stencil_decl,
    SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements,
    bool &overlap_eligible,
    int &overlap_width,
    vector<SgIntVal*> &overlap_flags);
  void BuildLoadNeighborStatements(
      SgExpression &grid_var,
      StencilRange &sr,
      SgExpression &reuse,
      bool is_periodic,
      SgStatementPtrList &statements,
      int &overlap_width,
      vector<SgIntVal*> &overlap_flags);
  void BuildLoadSubgridStatements(
      SgExpression &grid_var,
      StencilRange &sr,
      SgExpression &reuse,
      bool is_periodic,
      SgStatementPtrList &statements);
};

// REFACTORING: These functions should be moved inside MPIRuntimeBuilder
SgFunctionCallExp *BuildCallLoadSubgrid(SgExpression &grid_var,
                                        SgVariableDeclaration &grid_range,
                                        SgExpression &reuse);
SgFunctionCallExp *BuildCallLoadSubgridUniqueDim(SgExpression &grid_var,
                                                 StencilRange &sr,
                                                 SgExpression &reuse);

SgFunctionCallExp *BuildLoadNeighbor(SgExpression &grid_var,
                                     StencilRange &sr,
                                     SgScopeStatement &scope,
                                     SgExpression &reuse,
                                     SgExpression &overlap,
                                     bool is_periodic);
SgFunctionCallExp *BuildActivateRemoteGrid(SgExpression *grid_var,
                                           bool active);

                                   


} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_ */
