// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_OPENCL_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_MPI_OPENCL_TRANSLATOR_H_

#include "translator/translator.h"
#include "translator/translator_common.h"
#include "translator/mpi_translator.h"
#include "translator/opencl_translator.h"

namespace physis {
namespace translator {

class MPIOpenCLOptimizer;

class MPIOpenCLTranslator: public MPITranslator {
 protected:
  //! Used to generate OpenCL-related code.
  OpenCLTranslator *opencl_trans_;
  //! Optimization flag to enable the multi-stream boundary processing.
  bool flag_multistream_boundary_;
  string boundary_kernel_width_name_;
  string inner_prefix_;
  string boundary_suffix_;
  std::set<SgFunctionSymbol*> cache_config_done_;  
 public:
  MPIOpenCLTranslator(const Configuration &config);
  virtual ~MPIOpenCLTranslator();
  friend class MPIOpenCLOptimizer;
  virtual void SetUp(SgProject *project, TranslationContext *context);
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
      SgExpression *dom_ref, SgExpression *width,
      SgStatement *ifclause) const;
  virtual void ProcessStencilMap(StencilMap *smap, SgVarRefExp *stencils,
                                 int stencil_index, Run *run,
                                 SgScopeStatement *function_body,
                                 SgScopeStatement *loop_body,
                                 SgVariableDeclaration *dec_local_size,
                                 SgVariableDeclaration *argc_idx
                                 );
  virtual void BuildRunBody(SgBasicBlock *block, Run *run,
                            SgFunctionDeclaration *run_func);
  virtual void translateKernelDeclaration(SgFunctionDeclaration *node);
  //! Generates a OpenCL function declaration that runs a stencil map. 
  /*!
    \param s The stencil map object.
    \return The function declaration.
  */
  virtual SgFunctionDeclaration *BuildRunKernel(StencilMap *s);
  //! A helper function for BuildRunKernel.
  /*!
    \param stencil The stencil map object.
    \param dom_arg The stencil domain.
    \return The body of the run function.
  */
  virtual SgBasicBlock *BuildRunKernelBody(
      StencilMap *stencil, SgInitializedName *dom_arg);
  //! Generates OpenCL functions that run a boundary stencil.
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
    \param dom_arg The whole domain.
    \return The basic block.
  */
  virtual SgBasicBlock *BuildRunBoundaryKernelBody(
      StencilMap *stencil, SgInitializedName *dom_arg);
  //! Generates OpenCL functions that run boundary stencils.
  /*!
    This is used to generate calls to multiple boundary kernels.
    
    \param s The stencil map object.
    \return The function declarations.
  */
  virtual SgFunctionDeclarationPtrVector BuildRunMultiStreamBoundaryKernel(
      StencilMap *s);
  virtual SgBasicBlock *BuildRunMultiStreamBoundaryKernelBody(
      StencilMap *stencil, SgInitializedName *grid_arg,
      SgInitializedName *dom_arg, int dim, bool fw);
  //! Generates a OpenCL function executing an interior stencil.
  /*!
    \param s The stencil map.
    \return The OpenCL function with a call to the interior stencil. 
  */
  virtual SgFunctionDeclaration *BuildRunInteriorKernel(StencilMap *s);
  //! Generates code to run an inner stencil kernel.
  /*!
    A helper function for BuildRunInteriorKernel.
    
    \param stencil a stencil map object to generate calls.
    \param dom_arg the domain argument for the stencil map.
    \return The basic block containing the generated code.
  */
  virtual SgBasicBlock *BuildRunInteriorKernelBody(
      StencilMap *stencil,  SgInitializedName *dom_arg);
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
  virtual bool translateGetKernel(SgFunctionCallExp *node,
                                  SgInitializedName *gv,
                                  bool is_periodic);
  void BuildFunctionParamList(SgClassDefinition *param_struct_def,
                              SgFunctionParameterList *&params,
                              SgInitializedName *&grid_arg,
                              SgInitializedName *&dom_arg);

 protected:
  std::string kernel_mode_macro() const { return "PHYSIS_MPI_OPENCL_KERNEL_MODE"; } ;


  virtual SgType *BuildOnDeviceGridType(GridType *gt, int inner_device);
  virtual SgType *BuildOnDeviceDomType(void);

  virtual SgVariableDeclaration *BuildGridSizeDeclaration(
      const SgName &name,
      SgExpression *dom_dim_x, SgExpression *dom_dim_y,      
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL);
  virtual SgIfStmt *BuildDomainInclusionCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgExpression *dom_ref) const;
  virtual void Define_and_Construct_griddom_in_device(
      StencilMap *stencil,
      SgInitializedName *dom_arg,
      SgScopeStatement *scope
                                                      );
  virtual SgExprListExp* BuildKernelCallArgList(
      StencilMap *stencil, SgExpressionPtrList &index_args);
  virtual SgFunctionCallExp* BuildKernelCall(
      StencilMap *stencil, SgExpressionPtrList &index_args);
  virtual void add_macro_mark_function_inner_device(SgFunctionDeclaration *func);

 public:
  SgFunctionCallExp *BuildGridGetDev(SgExpression *grid_var);
  SgFunctionCallExp *BuildGetLocalSize(SgExpression *dim);
  SgFunctionCallExp *BuildGetLocalOffset(SgExpression *dim);
  SgFunctionCallExp *BuildDomainShrink(SgExpression *dom,
                                       SgExpression *width);
  SgFunctionCallExp *BuildCLThreadSynchronize(void);
  SgExpression *BuildStreamBoundaryKernel(int idx);

 public:
  virtual void add_opencl_extension_pragma();

 protected:
  virtual void FixAST() {}

};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_OPENCL_TRANSLATOR_H_ */
