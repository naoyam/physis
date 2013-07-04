#ifndef PHYSIS_TRANSLATOR_OPENCL_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_OPENCL_TRANSLATOR_H_
#undef USING_OLD_SOURCE

#include <string>
#include "translator/reference_translator.h"

#define DEBUG_FIX_AST_APPEND  1
#define DEBUG_FIX_DUP_SGEXP   1
#define DEBUG_FIX_CONSISTENCY 1

namespace physis {
namespace translator {

class OpenCLTranslator : public ReferenceTranslator {
 public:
  OpenCLTranslator(const Configuration &config);

 protected:  
  int block_dim_x_;
  int block_dim_y_;
  int block_dim_z_;

  std::string kernel_mode_macro() { return "PHYSIS_OPENCL_KERNEL_MODE"; } ;

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

  virtual void check_consistency(void);
  virtual void check_consistency_variable(void);
  virtual void check_consistency_function_definition(void);

  void Finish();

  virtual void translateKernelDeclaration(SgFunctionDeclaration *node);
  virtual SgExpression *buildOffset(
      SgInitializedName *gv,
      SgScopeStatement *scope,
      int numDim,
      const StencilIndexList *sil,
      bool is_periodic,
      SgExpressionPtrList &args);
  virtual void translateGet(
      SgFunctionCallExp *node,
      SgInitializedName *gv,
      bool isKernel, bool is_periodic);
  virtual void translateEmit(SgFunctionCallExp *node, SgInitializedName *gv);

  // Nothing performed for this target for now
  virtual void FixAST() {}

 public:
  virtual SgVariableDeclaration *generate2DLocalsize(
      std::string name_var,
      SgExpression *block_dimx, SgExpression *block_dimy,
      SgScopeStatement *scope);
  virtual SgVariableDeclaration *generate2DGlobalsize(
      std::string name_var,
      SgExpression *stencil_var,
      SgExpression *block_dimx, SgExpression *block_dimy,
      SgScopeStatement *scope);


 protected:
  virtual void BuildRunBody(SgBasicBlock *block, Run *run,
                            SgFunctionDeclaration *run_func);
 public:
  virtual SgBasicBlock *block_setkernelarg(
      SgVariableDeclaration *argc_idx,
      SgAssignInitializer *sginit,
      SgType *sgtype = 0);
 protected:
  virtual SgBasicBlock *GenerateRunLoopBody(
      SgScopeStatement *outer_block,      
      Run *run,
      SgFunctionDeclaration *run_func);
 public:
  virtual SgExpression *BuildBlockDimX();
  virtual SgExpression *BuildBlockDimY();
  virtual SgExpression *BuildBlockDimZ();
 protected:
  virtual SgType *BuildOnDeviceGridType(GridType *gt);


  virtual SgFunctionDeclaration *BuildRunKernel(StencilMap *s);
  virtual SgBasicBlock *generateRunKernelBody(
      StencilMap *stencil,
      SgInitializedName *grid_arg,
      SgInitializedName *dom_arg);
  virtual SgFunctionCallExp* generateKernelCall(
      StencilMap *stencil,
      SgExpressionPtrList &indexArgs,
      SgScopeStatement *containingScope);
  virtual SgIfStmt *BuildDomainInclusionCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgExpression *dom_ref);


  virtual void arg_add_grid_type(
      SgFunctionParameterList *args, SgType *type,
      int num_dim, int num_pos_grid);
  virtual void arg_add_grid_type(
      SgFunctionParameterList *args, SgInitializedName *arg);
  virtual void arg_add_grid_type(
      SgFunctionParameterList *args, std::string name_arg,
      std::string name_type, int num_dim);
  virtual void arg_add_dom_type(SgFunctionParameterList *args, int num_dim);
  virtual SgExprListExp * new_arguments_of_funccall_in_device(
      SgExprListExp *argexplist, SgScopeStatement *scope = NULL);


  virtual std::string name_var_dom(int dim, int maxflag);
  virtual std::string name_var_grid(int offset, std::string suffix);
  virtual std::string name_var_grid_dim(int offset, int dim);
  virtual std::string name_var_grid_dim(std::string oldname, int dim);
  virtual std::string name_var_gridptr(int offset) { return name_var_grid(offset, "buf"); };
  virtual std::string name_var_gridptr(std::string oldname) { return "__PS_" + oldname + "_buf"; };
  virtual std::string name_var_gridattr(int offset) { return name_var_grid(offset, "attr"); };
  virtual std::string name_var_gridattr(std::string oldname) { return "__PS_" + oldname + "_attr"; };
 public:
  virtual std::string name_new_kernel(std::string oldname) { return "__PS_opencl_" + oldname; };

 public:
  virtual void add_opencl_extension_pragma();

}; // class OpenCLTranslator: public ReferenceTranslator 


} // namespace translator
} // namespace physis

#endif /* define PHYSIS_TRANSLATOR_OPENCL_TRANSLATOR_H_ */
