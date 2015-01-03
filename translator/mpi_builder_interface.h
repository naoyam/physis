// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_MPI_BUILDER_INTERFACE_H_
#define PHYSIS_MPI_BUILDER_INTERFACE_H_

#include "translator/translator_common.h"

namespace physis {
namespace translator {

//! Pure interface class defining builder methods
class MPIBuilderInterface {
 public:
  virtual ~MPIBuilderInterface() {}

  //! Build a code sequence to load remote region necessary for a
  //! stencil map.
  /*!
    \param smap Load data for this stencil map
    \param stencil_decl Stencil variable declaration
    \param run Run object
    \param remote_grids Remote grid objects to hold remote data
    \param statements Output variable to hold generated statements
    \param overlap_eligible Output flag to indicate eligibility of overlapping
    \param overlap_width Width of overlapping stencil 
   */
  virtual void BuildLoadRemoteGridRegion(
      StencilMap *smap, SgVariableDeclaration *stencil_decl,
      Run *run, SgInitializedNamePtrList &remote_grids,
      SgStatementPtrList &statements, bool &overlap_eligible,
      int &overlap_width) = 0;

  //! Build statements to deactivate locally-cached remote grids
  /*!
    Remote subgrid caching may not be enabled on the backend runtime.
    
    \param smap
    \param stencil_decl
    \param scope
    \param remote_grids
   */
  virtual void BuildDeactivateRemoteGrids(
      StencilMap *smap,
      SgVariableDeclaration *stencil_decl,      
      const SgInitializedNamePtrList &remote_grids,
      SgStatementPtrList &stmt_list) = 0;
  
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_MPI_RUNTIME_BUILDER_H_ */
