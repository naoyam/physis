// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_TRANSLATOR_H_

#include "translator/translator.h"
#include "translator/translator_common.h"
#include "translator/mpi_translator.h"
#include "translator/cuda_translator.h"
#include "translator/mpi_cuda_runtime_builder.h"

namespace physis {
namespace translator {

class MPICUDATranslator: public MPITranslator {
 protected:
  //! Used to generate CUDA-related code.
  // Note: Is this dependency really non avoidable? No, it can be
  // avoided if the config logic is duplicated. Seems just fine to
  // delegate to the CUDA transaltor.
  CUDATranslator *cuda_trans_;
  //! Optimization flag to enable the multi-stream boundary processing.
  bool flag_multistream_boundary_;
  string boundary_kernel_width_name_;
  string inner_prefix_;
  string boundary_suffix_;
  std::set<SgFunctionSymbol*> cache_config_done_;
  virtual void FixAST();
  virtual MPICUDARuntimeBuilder *builder() {
    return dynamic_cast<MPICUDARuntimeBuilder*>(rt_builder_);
  }
 public:
  MPICUDATranslator(const Configuration &config);
  virtual ~MPICUDATranslator();
  virtual void SetUp(SgProject *project, TranslationContext *context,
                     BuilderInterface *rt_builder);
  virtual void Finish();
  //! Generates an IF block to exclude indices outside an inner domain.
  /*!
    \param indices the indices to check.
    \param dom_ref the domain.
    \param width the halo width.
    \param ifclause the IF clause to be taken when the condition is true.
    \return The IF block.
   */
  virtual SgIfStmt *BuildDomainInclusionInnerCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgInitializedName *dom_ref, SgExpression *width,
      SgStatement *ifclause) const;
  virtual void ProcessStencilMap(StencilMap *smap, 
                                 int stencil_index, Run *run,
                                 SgScopeStatement *function_body,
                                 SgScopeStatement *loop_body);

  virtual void BuildRunBody(SgBasicBlock *block, Run *run,
                            SgFunctionDeclaration *run_func);
  //! Generates a basic block of the run loop body.
  /*!
    This is a helper function for BuildRunBody. The run parameter
    contains stencil kernel calls and the number of iteration. This
    function generates a sequence of code to call the stencil kernels,
    which is then included in the for loop that iterates the given
    number of times. 
    
    \param run The stencil run object.
    \return outer_block The outer block where the for loop is included.
   */
  virtual SgBasicBlock *BuildRunLoopBody(Run *run,
                                         SgScopeStatement *outer_block);
  virtual void TranslateKernelDeclaration(SgFunctionDeclaration *node);
  //! Generates CUDA functions that run a boundary stencil.
  /*!
    \param s The stencil map object.
    \return The function declarations.
   */
  virtual SgFunctionDeclarationPtrVector BuildRunBoundaryKernel(
      StencilMap *s);
  //! Generates a basic block to run the boundary kernel of a stencil.
  /*!
    A helper function for BuildRunBoundaryKernel.
    
    \param stencil The stencil map object.
    \param param Parameter list for the run function
    \return The basic block.
   */
  virtual SgBasicBlock *BuildRunBoundaryKernelBody(
      StencilMap *stencil, SgFunctionParameterList *param);
  //! Generates CUDA functions that run boundary stencils.
  /*!
    This is used to generate calls to multiple boundary kernels.
    
    \param s The stencil map object.
    \return The function declarations.
   */
  virtual SgFunctionDeclarationPtrVector BuildRunMultiStreamBoundaryKernel(
      StencilMap *s);
  virtual SgBasicBlock *BuildRunMultiStreamBoundaryKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg, SgFunctionParameterList *params,
      int dim, bool fw);
  //! Generates a CUDA function executing an interior stencil.
  /*!
    \param s The stencil map.
    \return The CUDA function with a call to the interior stencil. 
   */
  virtual SgFunctionDeclaration *BuildRunInteriorKernel(StencilMap *s);
  //! Generates code to run an inner stencil kernel.
  /*!
    A helper function for BuildRunInteriorKernel.
    
    \param stencil a stencil map object to generate calls.
    \param param Parameters for the run function
    \return The basic block containing the generated code.
  */
  virtual SgBasicBlock *BuildRunInteriorKernelBody(
      StencilMap *stencil,  SgFunctionParameterList *param);
  //! Generates a kernel declaration optimized for interior domains.
  /*
    \param original The original kernel declaration.
    \return The interior kernel.
   */
  virtual SgFunctionDeclaration *BuildInteriorKernel(
      SgFunctionDeclaration *original) const;
  //! Generates kernel declarations for boundaries.
  /*
    \param original The original kernel declaration.
    \return The boundary kernel list.
   */
  virtual SgFunctionDeclarationPtrVector
  BuildBoundaryKernel(SgFunctionDeclaration *original);  
  std::string GetBoundarySuffix(int dim, bool fw);
  std::string GetBoundarySuffix();
#if 0  // use the parent class implementation
  virtual bool TranslateGetKernel(SgFunctionCallExp *node,
                                  SgInitializedName *gv,
                                  bool is_periodic);
#endif   
  void BuildFunctionParamList(SgClassDefinition *param_struct_def,
                              SgFunctionParameterList *&params,
                              SgInitializedName *&grid_arg,
                              SgInitializedName *&dom_arg);

  //! Set the cache config of all functions related to fs
  void SetCacheConfig(StencilMap *smap, SgFunctionSymbol *fs,
                      SgScopeStatement *function_body,
                      bool overlap_enabled);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_TRANSLATOR_H_ */
