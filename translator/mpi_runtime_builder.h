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

  //! Build a code sequence to load remote region necessary for a grid
  /*!
    This is a helper function for
    BuildLoadRemoteGridRegion(StencilMap*, SgVariableDeclaration*,...)

    \param grid_param Grid variable    
    \param smap Load data for this stencil map
    \param stencil_decl Stencil variable declaration
    \param remote_grids Remote grid objects to hold remote data
    \param statements Output variable to hold generated statements
    \param overlap_eligible Output flag to indicate eligibility of overlapping
    \param overlap_width Width of overlapping stencil
    \param overlap_flags Overlap flag variables
   */
  virtual void BuildLoadRemoteGridRegion(
    SgInitializedName &grid_param,
    StencilMap &smap,
    SgVariableDeclaration &stencil_decl,
    SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements,
    bool &overlap_eligible,
    int &overlap_width,
    vector<SgIntVal*> &overlap_flags);
  //! Build a code sequence to load remote region necessary for a grid member
  /*!
    This is a helper function for
    BuildLoadRemoteGridRegion(SgInitializedName &, StencilMap &,...)

    If member_index is below zero, halo for the whole struct is fetched.

    \param grid_param Grid variable
    \param member_index Zero-based index of the member
    \param sr StencilRange for the member
    \param smap Load data for this stencil map
    \param stencil_decl Stencil variable declaration
    \param remote_grids Remote grid objects to hold remote data
    \param statements Output variable to hold generated statements
    \param overlap_eligible Output flag to indicate eligibility of overlapping
    \param overlap_width Width of overlapping stencil
    \param overlap_flags Overlap flag variables
   */
  virtual void BuildLoadRemoteGridRegion(
    SgInitializedName &grid_param,
    int member_index,
    StencilRange &sr,
    StencilMap &smap,
    SgVariableDeclaration &stencil_decl,
    SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements,
    bool &overlap_eligible,
    int &overlap_width,
    vector<SgIntVal*> &overlap_flags);
  //! Build a code sequence to call loadNeighbor for a grid member
  /*!
    This is a helper function for
    BuildLoadRemoteGridRegion(SgInitializedName &, int, StencilRange&,...)

    \param grid_var Grid variable
    \param member_index Zero-based index of the member
    \param sr StencilRange for the member
    \param reuse Flag expression to indicate reuse
    \param is_periodic flag for periodic boundary condition
    \param statements Output variable to hold generated statements
    \param overlap_width Width of overlapping stencil
    \param overlap_flags Overlap flag variables
  */
  virtual void BuildLoadNeighborStatements(
      SgExpression &grid_var,
      int member_index,
      StencilRange &sr,
      SgExpression &reuse,
      bool is_periodic,
      SgStatementPtrList &statements,
      int &overlap_width,
      vector<SgIntVal*> &overlap_flags);

  virtual SgFunctionCallExp *BuildLoadNeighbor(
      SgExpression &grid_var,
      int member_index,
      StencilRange &sr,
      SgScopeStatement &scope,
      SgExpression &reuse,
      SgExpression &overlap,
      bool is_periodic);
  
  //! Build a code sequence to call loadSubgrid for a grid member
  /*!
    This is a helper function for
    BuildLoadRemoteGridRegion(SgInitializedName &, int,
    StencilRange&,...)
    
    \param grid_var Grid variable
    \param sr StencilRange for the member
    \param reuse Flag expression to indicate reuse
    \param is_periodic flag for periodic boundary condition
    \param statements Output variable to hold generated statements
  */
  virtual void BuildLoadSubgridStatements(
      SgExpression &grid_var,
      StencilRange &sr,
      SgExpression &reuse,
      bool is_periodic,
      SgStatementPtrList &statements);

  virtual SgFunctionCallExp *BuildCallLoadSubgrid(
      SgExpression &grid_var,
      SgVariableDeclaration &grid_range,
      SgExpression &reuse);

  virtual SgFunctionCallExp *BuildCallLoadSubgridUniqueDim(
      SgExpression &grid_var,
      StencilRange &sr,
      SgExpression &reuse);
  
  virtual SgFunctionCallExp *BuildActivateRemoteGrid(
      SgExpression *grid_var,
      bool active);
};


} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_ */
