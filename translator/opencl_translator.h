#ifndef PHYSIS_TRANSLATOR_OPENCL_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_OPENCL_TRANSLATOR_H_
#undef USING_OLD_SOURCE

#include <string>
#include "translator/reference_translator.h"

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
#ifdef USING_OLD_SOURCE
        void run();
#else
        void Finish();
#endif

        virtual void translateKernelDeclaration(SgFunctionDeclaration *node);
        virtual SgExpression *buildOffset(
                    SgInitializedName *gv,
                    SgScopeStatement *scope,
                    int numDim,
                    SgExpressionPtrList &args);
        virtual void translateGet(
                  SgFunctionCallExp *node,
                  SgInitializedName *gv,
                  bool isKernel);
        virtual void translateEmit(SgFunctionCallExp *node, SgInitializedName *gv);


        virtual SgVariableDeclaration *generate2DLocalsize(
            std::string name_var,
            SgExpression *block_dimx, SgExpression *block_dimy,
            SgScopeStatement *scope);
        virtual SgVariableDeclaration *generate2DGlobalsize(
            std::string name_var,
            SgExpression *stencil_var,
            SgExpression *block_dimx, SgExpression *block_dimy,
            SgScopeStatement *scope);


        virtual SgBasicBlock *BuildRunBody(Run *run);
        virtual SgBasicBlock *block_setkernelarg(
            SgVariableDeclaration *argc_idx,
            SgAssignInitializer *sginit);
        virtual SgBasicBlock *GenerateRunLoopBody(
                  Run *run,
                  SgScopeStatement *outer_block);


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


        virtual SgExpression *BuildBlockDimX();
        virtual SgExpression *BuildBlockDimY();
        virtual SgExpression *BuildBlockDimZ();
        virtual SgType *BuildOnDeviceGridType(GridType *gt);


        virtual void arg_add_grid_type(
                      SgFunctionParameterList *args, SgType *type,
                      int num_dim, int num_pos_grid);
        virtual void arg_add_grid_type(
                      SgFunctionParameterList *args, SgInitializedName *arg);
        virtual void arg_add_grid_type(
                      SgFunctionParameterList *args, std::string name_arg,
                      std::string name_type, int num_dim);
        virtual void arg_add_dom_type(SgFunctionParameterList *args, int num_dim);
        virtual SgExprListExp * new_arguments_of_funccall_in_device(SgExprListExp *argexplist);


        virtual std::string name_var_dom(int dim, int maxflag);
        virtual std::string name_var_grid(int offset, std::string suffix);
        virtual std::string name_var_grid_dim(int offset, int dim);
        virtual std::string name_var_grid_dim(std::string oldname, int dim);
        virtual std::string name_var_gridptr(int offset) { return name_var_grid(offset, "buf"); };
        virtual std::string name_var_gridptr(std::string oldname) { return "__PS_" + oldname + "_buf"; };
        virtual std::string name_var_gridattr(int offset) { return name_var_grid(offset, "attr"); };
        virtual std::string name_var_gridattr(std::string oldname) { return "__PS_" + oldname + "_attr"; };
        virtual std::string name_new_kernel(std::string oldname) { return "__PS_opencl_" + oldname; };

    }; // class OpenCLTranslator: public ReferenceTranslator 


  } // namespace translator
} // namespace physis

#endif /* define PHYSIS_TRANSLATOR_OPENCL_TRANSLATOR_H_ */
