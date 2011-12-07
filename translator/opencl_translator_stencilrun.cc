#include "translator/opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/SageBuilderEx.h"

namespace sbx = physis::translator::SageBuilderEx;
namespace si = SageInterface;
namespace sb = SageBuilder;

#define BLOCK_DIM_X_DEFAULT (64)
#define BLOCK_DIM_Y_DEFAULT (4)
#define BLOCK_DIM_Z_DEFAULT (1)

namespace physis {
namespace translator {

// BuildRunBody:
// Genereate the skeleton of loop for iteration
SgBasicBlock *OpenCLTranslator::BuildRunBody(Run *run)
{
  // build loop block
  SgBasicBlock *block = sb::buildBasicBlock();

  // declare "int i" in the block;
  SgVariableDeclaration *loop_index =
      sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  block->append_statement(loop_index);

  // i = 0;
  SgStatement *loop_init =
      sb::buildAssignStatement(sb::buildVarRefExp(loop_index),
                               sb::buildIntVal(0));
  // i < iter
  SgStatement *loop_test =
      sb::buildExprStatement(
          sb::buildLessThanOp(sb::buildVarRefExp(loop_index),
                              sb::buildVarRefExp("iter", block)));
  // ++i
  SgExpression *loop_incr =
      sb::buildPlusPlusOp(sb::buildVarRefExp(loop_index));

  // Generate loop body
  SgBasicBlock *loop_body = GenerateRunLoopBody(run, block);
  SgForStatement *loop =
      sb::buildForStatement(loop_init, loop_test, loop_incr, loop_body);

  // Add TranceStencilRun
  TraceStencilRun(run, loop, block);
  
  return block;
} // generateRunBody


// block_setkernelarg:
//
// Create the block in the loop in __PSStencilRun like
// { long int j = <sginit>; __PSSetKernelArg(argc, sizeof(j), &j); argc++; }
//
SgBasicBlock *OpenCLTranslator::block_setkernelarg(
    SgVariableDeclaration *argc_idx,
    SgAssignInitializer *sginit
)
{
    // Initialize block
    SgBasicBlock *block_ret = sb::buildBasicBlock();

    // long int j;
    SgVariableDeclaration *j_idx = 
       sb::buildVariableDeclaration("j", sb::buildLongType(), NULL, block_ret);
    block_ret->append_statement(j_idx);
    // j = <sginit>;
    j_idx->reset_initializer(sginit);

    {
      // argument initialize
      SgExprListExp* args = sb::buildExprListExp();
      // argc
      args->append_expression(sb::buildVarRefExp(argc_idx));
      // sizeof(j)
      {
        SgSizeOfOp *exp_call_sizeof = sb::buildSizeOfOp(sb::buildVarRefExp(j_idx));
          args->append_expression(exp_call_sizeof);
      }
     // &j
      {
        SgExpression *exp_ptr_j = sb::buildAddressOfOp(sb::buildVarRefExp(j_idx));
          args->append_expression(exp_ptr_j);
      }
      // __PSSetKernelArg(argc, sizeof(j), &j);
      SgFunctionCallExp *setkernelarg_exp =
        sb::buildFunctionCallExp(
          sb::buildFunctionRefExp("__PSSetKernelArg"),
            args
              );
      block_ret->append_statement(sb::buildExprStatement(setkernelarg_exp));
    }

    // ++argc;
    {
      SgExpression *exp_plusplus_j = sb::buildPlusPlusOp(sb::buildVarRefExp(argc_idx));
      block_ret->append_statement(sb::buildExprStatement(exp_plusplus_j));
    }

    // ret
    return block_ret;

} // block_setkernelarg

// generate2DLocalsize
//
// declare localsize used in OpenCL:
// size_t localsize[] = {dimx, dimy, 1};
SgVariableDeclaration *OpenCLTranslator::generate2DLocalsize(
    std::string name_var,
    SgExpression *block_dimx, SgExpression *block_dimy,
    SgScopeStatement *scope)
{
    SgVariableDeclaration *sg_lc;
    // TODO: use ivec_type_ ?
    // size_t name_var[]
    SgType *type_sg_lc =
      sb::buildArrayType(
        sb::buildOpaqueType("size_t", scope)
          );
    // Create array {dimx, dimy, 1}
    SgExprListExp *def_lc = sb::buildExprListExp();
    def_lc->append_expression(block_dimx);
    def_lc->append_expression(block_dimy);
    def_lc->append_expression(sb::buildIntVal(1));

    SgAggregateInitializer *initval_lc =
      sb::buildAggregateInitializer(def_lc, type_sg_lc);

    // size_t name_var[] = {dimx, dimy, 1};
    sg_lc = sb::buildVariableDeclaration(
      name_var, type_sg_lc, initval_lc, scope);

    return sg_lc;
} // generate2DLocalsize

// generate2DGlobalsize
//
// declare globalsize used in OpenCL:
// size_t global[] = {(size_t)(ceil(((float)(s0.dom.local_max[0])) / dim_x) * dim_x),
//    (size_t)(ceil(((float)(s0.dom.local_max[1]) / dim_y)) * dim_y),
//    1};
//
SgVariableDeclaration *OpenCLTranslator::generate2DGlobalsize(
    std::string name_var,
    SgExpression *stencil_var,
    SgExpression *block_dimx, SgExpression *block_dimy,
    SgScopeStatement *scope)
{
    SgVariableDeclaration *sg_gb;
    // TODO: use ivec_type_ ?
    // size_t name_var[]
    SgType *type_sg_gb =
      sb::buildArrayType(
        sb::buildOpaqueType("size_t", scope)
          );
    // Create array
    SgExprListExp *def_gb = sb::buildExprListExp();

    int case_xy;
    for (case_xy = 0; case_xy <= 1; case_xy++){
      SgExpression *block_dimxy;
      switch(case_xy) {
        case 0:
          block_dimxy = block_dimx;
          break;
        case 1:
          block_dimxy = block_dimy;
          break;
        default:
          block_dimxy = NULL;
          break;
      } // switch(case_xy)

      // (size_t)(ceil(((float)(s0.dom.local_max[xy])) / dimxy) * dimxy)
      SgExpression *gb_xy =
        sb::buildCastExp(
          sb::buildMultiplyOp(
            sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("ceil"),
              sb::buildExprListExp(
                sb::buildDivideOp(
                  sb::buildCastExp(
                    BuildStencilDomMaxRef(stencil_var, case_xy),
                    sb::buildFloatType()
                      ),
                  block_dimxy
                    )
                  )
                ),
            block_dimxy
              ),
          sb::buildOpaqueType("size_t", scope)
            );

      def_gb->append_expression(gb_xy);
    } // for (case_xy = 0; case_xy <= 1; case_xy++)
    def_gb->append_expression(sb::buildIntVal(1));

    SgAggregateInitializer *initval_gb =
      sb::buildAggregateInitializer(def_gb, type_sg_gb);

    // size_t name_var[] = {initval_gb};
    sg_gb = sb::buildVariableDeclaration(
      name_var, type_sg_gb, initval_gb, scope);

    return sg_gb;
} // generate2DLocalsize

// GenerateRunLoopBody:
//
// Generate the loop body where to call OpenCL kernel
SgBasicBlock *OpenCLTranslator::GenerateRunLoopBody(
    Run *run,
    SgScopeStatement *outer_block)
{
  // Generate the basic body
  SgBasicBlock *loop_body = sb::buildBasicBlock();

  // declare localsize
  SgVariableDeclaration *lc_idx =
    generate2DLocalsize(
        "localsize", BuildBlockDimX(), BuildBlockDimY(),
        outer_block);
  outer_block->append_statement(lc_idx);

  // Declare argc
  // with initialization: int argc = 0;
  SgVariableDeclaration *argc_idx =
      sb::buildVariableDeclaration("argc", sb::buildIntType(), NULL, loop_body);
  loop_body->append_statement(argc_idx);
  argc_idx->reset_initializer(
      sb::buildAssignInitializer(
          sb::buildIntVal(0)
          ));

  // Run each stencil kernels in sequence
  ENUMERATE(stencil_idx, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *sm = it->second;

    // Stencil kernel symbol
    SgFunctionSymbol *kern_sym =
        rose_util::getFunctionSymbol(sm->run());
    PSAssert(kern_sym);

    std::string stencil_name = "s" + toString(stencil_idx); // s0
    SgVarRefExp *stencil_var = sb::buildVarRefExp(stencil_name);

    // Stencil structure definition of StencilMap *sm
    SgClassDefinition *stencil_def = sm->GetStencilTypeDefinition();
    PSAssert(stencil_def);

    // declare globalsize_s0
    std::string name_gb = "globalsize_" + stencil_name;
    SgVariableDeclaration *gb_idx =
      generate2DGlobalsize(
        name_gb, stencil_var, BuildBlockDimX(), BuildBlockDimY(),
        outer_block);
    outer_block->append_statement(gb_idx);

    // __PSSetKernel("__PSStencilRun_kernel1");
    {
      SgExprListExp *setkernel_args = sb::buildExprListExp();
      // argument
      std::string kern_name = kern_sym->get_name().getString();
      SgStringVal *kern_name_sym = sb::buildStringVal(kern_name);
      setkernel_args->append_expression(kern_name_sym);
      SgFunctionCallExp *setkernel_exp =
        sb::buildFunctionCallExp(
          sb::buildFunctionRefExp("__PSSetKernel"),
              setkernel_args
                );
      loop_body->append_statement(sb::buildExprStatement(setkernel_exp));
    }

    {
      // Dimension
      int num_dim = sm->getNumDim();
      int pos_dim;
      for (pos_dim = 0; pos_dim < num_dim; pos_dim++) {
        int maxflag = 0;
        for (maxflag = 0; maxflag <= 1; maxflag++) {
          // Create the init value of j
          // s0.dom.local_<min,max>[pos_dim]
          SgExpression *exp_dom_minmax_ref;
          if (maxflag == 0) // add s0.dom.local_min[pos_dim];
            exp_dom_minmax_ref = BuildStencilDomMinRef(stencil_var, pos_dim);
          else // s0.dom.local_max[pos_dim];
            exp_dom_minmax_ref = BuildStencilDomMaxRef(stencil_var, pos_dim);

          // { long int j = s0.dom.local_<min,max>[pos_dim]; 
          //   __PSSetKernelArg(argc, sizeof(j), &j); argc++; }
          loop_body->append_statement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(exp_dom_minmax_ref)
                ));

        } // for (maxflag = 0 ; maxflag <= 1; maxflag++)
      } // for (pos_dim = 0; pos_dim < num_dim; pos_dim++)

    }

    // Enumerate members of parameter struct
    { 

      const SgDeclarationStatementPtrList &members = stencil_def->get_members();
      FOREACH(member, members.begin(), members.end()) {
        // member_decl should be dom, g, __g_index, for example
        SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
        // s0.dom, s0.g, and etc (dom, g: each stencil member)
        SgExpression *arg_sg =
          sb::buildDotExp(stencil_var, sb::buildVarRefExp(member_decl));

        // find s0.g (grid type)
        const SgInitializedNamePtrList &vars = member_decl->get_variables();
        GridType *gt = tx_->findGridType(vars[0]->get_type());
        if (!gt) continue;

        // Dimension
        int num_dim = sm->getNumDim();
        int pos_dim;
        for (pos_dim = 0; pos_dim < num_dim; pos_dim++){
          // Create the initial value of j : __PSGridDimDev(s0.g, pos_dim);
          SgExprListExp *args = sb::buildExprListExp();
          // s0.g
          args->append_expression(arg_sg);
          // pos_dum
          args->append_expression(sb::buildIntVal(pos_dim));
          // __PSGridDimDev(s0.g, pos_dim);
          SgFunctionCallExp *exp_call_GridDimDev =
            sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSGridDimDev"),
              args
                );
          // { long int j = __PSGridDimDev(s0.g, num_pos); 
          //   __PSSetKernelArg(argc, sizeof(j), &j); argc++; }
          loop_body->append_statement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(exp_call_GridDimDev)
                ));
        } // for (pos_dim = 0; pos_dim < num_dim; pos_dim++)

        {
          // create inner block
          SgBasicBlock *block_gbuf = sb::buildBasicBlock();
          // Create variable void *buf;
          SgVariableDeclaration *buf_idx =
            sb::buildVariableDeclaration(
              "buf", sb::buildPointerType(sb::buildVoidType()), NULL, block_gbuf);
          block_gbuf->append_statement(buf_idx);
          // buf = s0.g->buf;
          buf_idx->reset_initializer(
            sb::buildAssignInitializer(
              sb::buildArrowExp(arg_sg, sb::buildVarRefExp("buf"))
                ));
          {
            // Initialize argument
            SgExprListExp *args = sb::buildExprListExp();
            // argc
            args->append_expression(sb::buildVarRefExp(argc_idx));
            // (void *)(&buf)
            SgExpression *exp_ptr_buf = 
              sb::buildAddressOfOp(sb::buildVarRefExp(buf_idx));
            SgExpression *exp_cast_ptr_buf =
              sb::buildCastExp(
                  exp_ptr_buf,
                  sb::buildPointerType(sb::buildVoidType())
                  );
            args->append_expression(exp_cast_ptr_buf);
            // __PSSetKernelArgCLMem(argc, (void *)(&buf))
            SgFunctionCallExp *exp_setmem =
              sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("__PSSetKernelArgCLMem"),
                args
                  );
            block_gbuf->append_statement(sb::buildExprStatement(exp_setmem));
          }
          {
            // ++argc;
            SgExpression *exp_plusplus_j = sb::buildPlusPlusOp(sb::buildVarRefExp(argc_idx));
            block_gbuf->append_statement(sb::buildExprStatement(exp_plusplus_j));
          }
          
          // append block_gbuf to loop_body
          loop_body->append_statement(block_gbuf);
        }

        {
          // { long int j = s0.g->gridattr; 
          //   __PSSetKernelArg(argc, sizeof(j), &j); argc++; }
          loop_body->append_statement(
            block_setkernelarg(
                argc_idx,
                sb::buildAssignInitializer(
                  sb::buildArrowExp(arg_sg, sb::buildVarRefExp("gridattr"))
                )));
        } // 
      } // FOREACH(member, members.begin(), members.end())
    }

    {
      // (globalsize, localsize)
      SgExprListExp *args_runkernel = sb::buildExprListExp(
        sb::buildVarRefExp(gb_idx),
        sb::buildVarRefExp(lc_idx)
        );
      // __PSRunKernel(globalsize, localsize)
      SgFunctionCallExp *exp_runkernel = sb::buildFunctionCallExp(
        sb::buildFunctionRefExp("__PSRunKernel"),
        args_runkernel
        );
      loop_body->append_statement(sb::buildExprStatement(exp_runkernel));
    }

    // Swap not used
#if 0
    appendGridSwap(sm, stencil_var, loop_body);
#endif

    // reset argc: argc = 0;
    SgExpression *argc_zero_lhs = sb::buildVarRefExp(argc_idx);
    SgExpression *argc_zero_rhs = sb::buildIntVal(0);
    SgExpression *argc_zero = sb::buildAssignOp(argc_zero_lhs, argc_zero_rhs);
    loop_body->append_statement(sb::buildExprStatement(argc_zero));

  } // NUMERATE(stencil_idx, it, run->stencils().begin(), run->stencils().end())

  return loop_body;

} // GenerateRunLoopBody



} // namespace translator
} // namespace physis
