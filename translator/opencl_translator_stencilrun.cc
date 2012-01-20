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
#ifdef DEBUG_FIX_AST_APPEND
  si::appendStatement(loop_index, block);
#else
  block->append_statement(loop_index);
#endif

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
    SgAssignInitializer *sginit,
    SgType *sgtype
)
{
    // Initialize block
    SgBasicBlock *block_ret = sb::buildBasicBlock();

    SgType *sgtype_j = sgtype;
    if (!sgtype_j)
      sgtype_j = sb::buildLongType();
    // TYPE j; (TYPE: default: Long)
    SgVariableDeclaration *j_idx = 
       sb::buildVariableDeclaration("j", sgtype_j, NULL, block_ret);
#ifdef DEBUG_FIX_AST_APPEND
    si::appendStatement(j_idx, block_ret);
#else
    block_ret->append_statement(j_idx);
#endif
    // j = <sginit>;
    j_idx->reset_initializer(sginit);

    {
      // argument initialize
      SgExprListExp* args = sb::buildExprListExp();
      // argc
#ifdef DEBUG_FIX_AST_APPEND
      si::appendExpression(args, sb::buildVarRefExp(argc_idx));
#else
      args->append_expression(sb::buildVarRefExp(argc_idx));
#endif
      // sizeof(j)
      {
        SgSizeOfOp *exp_call_sizeof = sb::buildSizeOfOp(sb::buildVarRefExp(j_idx));
#ifdef DEBUG_FIX_AST_APPEND
        si::appendExpression(args, exp_call_sizeof);
#else
        args->append_expression(exp_call_sizeof);
#endif
      }
     // &j
      {
        SgExpression *exp_ptr_j = sb::buildAddressOfOp(sb::buildVarRefExp(j_idx));
#ifdef DEBUG_FIX_AST_APPEND
        si::appendExpression(args, exp_ptr_j);
#else
        args->append_expression(exp_ptr_j);
#endif
      }
      // __PSSetKernelArg(argc, sizeof(j), &j);
      SgFunctionCallExp *setkernelarg_exp =
        sb::buildFunctionCallExp(
          sb::buildFunctionRefExp("__PSSetKernelArg"),
            args
              );
#ifdef DEBUG_FIX_AST_APPEND
      si::appendStatement(sb::buildExprStatement(setkernelarg_exp), block_ret);
#else
      block_ret->append_statement(sb::buildExprStatement(setkernelarg_exp));
#endif
    }

    // ++argc;
    {
      SgExpression *exp_plusplus_j = sb::buildPlusPlusOp(sb::buildVarRefExp(argc_idx));
#ifdef DEBUG_FIX_AST_APPEND
      si::appendStatement(sb::buildExprStatement(exp_plusplus_j), block_ret);
#else
      block_ret->append_statement(sb::buildExprStatement(exp_plusplus_j));
#endif
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
#ifdef DEBUG_FIX_AST_APPEND
    si::appendExpression(def_lc, block_dimx);
    si::appendExpression(def_lc, block_dimy);
    si::appendExpression(def_lc, sb::buildIntVal(1));
#else
    def_lc->append_expression(block_dimx);
    def_lc->append_expression(block_dimy);
    def_lc->append_expression(sb::buildIntVal(1));
#endif

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
#ifdef DEBUG_FIX_DUP_SGEXP
          block_dimxy = si::copyExpression(block_dimx);
#else
          block_dimxy = block_dimx;
#endif
          break;
        case 1:
#ifdef DEBUG_FIX_DUP_SGEXP
          block_dimxy = si::copyExpression(block_dimy);
#else
          block_dimxy = block_dimy;
#endif
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

#ifdef DEBUG_FIX_AST_APPEND
      si::appendExpression(def_gb, gb_xy);
#else
      def_gb->append_expression(gb_xy);
#endif
    } // for (case_xy = 0; case_xy <= 1; case_xy++)
#ifdef DEBUG_FIX_AST_APPEND
    si::appendExpression(def_gb, sb::buildIntVal(1));
#else
    def_gb->append_expression(sb::buildIntVal(1));
#endif

    SgAggregateInitializer *initval_gb =
      sb::buildAggregateInitializer(def_gb, type_sg_gb);

    // size_t name_var[] = {initval_gb};
    sg_gb = sb::buildVariableDeclaration(
      name_var, type_sg_gb, initval_gb, scope);

    // !! Note
    // This function calls "ceil", math.h needed
    LOG_INFO() << "Inserting <math.h> because of ceil call.\n";
    // TODO: The following does not work?
    // si::insertHeader("math.h", PreprocessingInfo::before, src_->get_globalScope());
    std::string str_insert = "";
    str_insert += "#ifndef "; str_insert += kernel_mode_macro();
    si::attachArbitraryText(src_->get_globalScope(), str_insert);
    si::attachArbitraryText(src_->get_globalScope(), "#include <math.h>");
    si::attachArbitraryText(src_->get_globalScope(), "#endif");

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
#ifdef DEBUG_FIX_AST_APPEND
  si::appendStatement(lc_idx, outer_block);
#else
  outer_block->append_statement(lc_idx);
#endif

  // Declare argc
  // with initialization: int argc = 0;
  SgVariableDeclaration *argc_idx =
      sb::buildVariableDeclaration("argc", sb::buildIntType(), NULL, loop_body);
#ifdef DEBUG_FIX_AST_APPEND
  si::appendStatement(argc_idx, loop_body);
#else
  loop_body->append_statement(argc_idx);
#endif
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
#ifdef DEBUG_FIX_AST_APPEND
    si::appendStatement(gb_idx, outer_block);
#else
    outer_block->append_statement(gb_idx);
#endif

    // __PSSetKernel("__PSStencilRun_kernel1");
    {
      SgExprListExp *setkernel_args = sb::buildExprListExp();
      // argument
      std::string kern_name = kern_sym->get_name().getString();
      SgStringVal *kern_name_sym = sb::buildStringVal(kern_name);
#ifdef DEBUG_FIX_AST_APPEND
      si::appendExpression(setkernel_args, kern_name_sym);
#else
      setkernel_args->append_expression(kern_name_sym);
#endif
      SgFunctionCallExp *setkernel_exp =
        sb::buildFunctionCallExp(
          sb::buildFunctionRefExp("__PSSetKernel"),
              setkernel_args
                );
#ifdef DEBUG_FIX_AST_APPEND
      si::appendStatement(
        sb::buildExprStatement(setkernel_exp), loop_body);
#else
      loop_body->append_statement(sb::buildExprStatement(setkernel_exp));
#endif
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
#ifdef DEBUG_FIX_DUP_SGEXP
          stencil_var = sb::buildVarRefExp(stencil_name);
#endif
          if (maxflag == 0) // add s0.dom.local_min[pos_dim];
            exp_dom_minmax_ref = BuildStencilDomMinRef(stencil_var, pos_dim);
          else // s0.dom.local_max[pos_dim];
            exp_dom_minmax_ref = BuildStencilDomMaxRef(stencil_var, pos_dim);

          // { long int j = s0.dom.local_<min,max>[pos_dim]; 
          //   __PSSetKernelArg(argc, sizeof(j), &j); argc++; }
#ifdef DEBUG_FIX_AST_APPEND
          si::appendStatement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(exp_dom_minmax_ref)
                ),
            loop_body);            
#else
          loop_body->append_statement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(exp_dom_minmax_ref)
                ));
#endif

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
#ifdef DEBUG_FIX_DUP_SGEXP
        stencil_var = sb::buildVarRefExp(stencil_name);
#endif
        SgExpression *arg_sg =
          sb::buildDotExp(stencil_var, sb::buildVarRefExp(member_decl));

        // find s0.g (grid type)
        const SgInitializedNamePtrList &vars = member_decl->get_variables();
        SgType *member_type = vars[0]->get_type();
        GridType *gt = tx_->findGridType(member_type);
        if (!gt) {
          if (Domain::isDomainType(member_type)) 
            continue; // Domain type already done

          // Add normal type
          si::appendStatement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(arg_sg), member_type
                ),
            loop_body);
          continue; // Done
        }


         // Handle grid type
        // Dimension
        int num_dim = sm->getNumDim();
        int pos_dim;
        for (pos_dim = 0; pos_dim < num_dim; pos_dim++){
          // Create the initial value of j : __PSGridDimDev(s0.g, pos_dim);
          SgExprListExp *args = sb::buildExprListExp();
          // s0.g
#ifdef DEBUG_FIX_DUP_SGEXP
          // Once more
          stencil_var = sb::buildVarRefExp(stencil_name);
          arg_sg =
              sb::buildDotExp(stencil_var, sb::buildVarRefExp(member_decl));
#endif
#ifdef DEBUG_FIX_AST_APPEND
          si::appendExpression(args, arg_sg);
#else
          args->append_expression(arg_sg);
#endif
          // pos_dum
#ifdef DEBUG_FIX_AST_APPEND
          si::appendExpression(args, sb::buildIntVal(pos_dim));
#else
          args->append_expression(sb::buildIntVal(pos_dim));
#endif
          // __PSGridDimDev(s0.g, pos_dim);
          SgFunctionCallExp *exp_call_GridDimDev =
            sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSGridDimDev"),
              args
                );
          // { long int j = __PSGridDimDev(s0.g, num_pos); 
          //   __PSSetKernelArg(argc, sizeof(j), &j); argc++; }

#ifdef DEBUG_FIX_AST_APPEND
          si::appendStatement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(exp_call_GridDimDev)
                ),
            loop_body);
#else
          loop_body->append_statement(
            block_setkernelarg(
              argc_idx, sb::buildAssignInitializer(exp_call_GridDimDev)
                ));
#endif
        } // for (pos_dim = 0; pos_dim < num_dim; pos_dim++)

        {
          // create inner block
          SgBasicBlock *block_gbuf = sb::buildBasicBlock();
          // Create variable void *buf;
          SgVariableDeclaration *buf_idx =
            sb::buildVariableDeclaration(
              "buf", sb::buildPointerType(sb::buildVoidType()), NULL, block_gbuf);
#ifdef DEBUG_FIX_AST_APPEND
          si::appendStatement(buf_idx, block_gbuf);
#else
          block_gbuf->append_statement(buf_idx);
#endif
          // buf = s0.g->buf;
          buf_idx->reset_initializer(
            sb::buildAssignInitializer(
              sb::buildArrowExp(arg_sg, sb::buildVarRefExp("buf"))
                ));
          {
            // Initialize argument
            SgExprListExp *args = sb::buildExprListExp();
            // argc
#ifdef DEBUG_FIX_AST_APPEND
            si::appendExpression(args, sb::buildVarRefExp(argc_idx));
#else
            args->append_expression(sb::buildVarRefExp(argc_idx));
#endif
            // (void *)(&buf)
            SgExpression *exp_ptr_buf = 
              sb::buildAddressOfOp(sb::buildVarRefExp(buf_idx));
            SgExpression *exp_cast_ptr_buf =
              sb::buildCastExp(
                  exp_ptr_buf,
                  sb::buildPointerType(sb::buildVoidType())
                  );
#ifdef DEBUG_FIX_AST_APPEND
            si::appendExpression(args, exp_cast_ptr_buf);
#else
            args->append_expression(exp_cast_ptr_buf);
#endif
            // __PSSetKernelArgCLMem(argc, (void *)(&buf))
            SgFunctionCallExp *exp_setmem =
              sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("__PSSetKernelArgCLMem"),
                args
                  );
#ifdef DEBUG_FIX_AST_APPEND
            si::appendStatement(sb::buildExprStatement(exp_setmem), block_gbuf);
#else
            block_gbuf->append_statement(sb::buildExprStatement(exp_setmem));
#endif
          }
          {
            // ++argc;
            SgExpression *exp_plusplus_j = sb::buildPlusPlusOp(sb::buildVarRefExp(argc_idx));
#ifdef DEBUG_FIX_AST_APPEND
            si::appendStatement(
              sb::buildExprStatement(exp_plusplus_j), block_gbuf);
#else
            block_gbuf->append_statement(sb::buildExprStatement(exp_plusplus_j));
#endif
          }
          
          // append block_gbuf to loop_body
#ifdef DEBUG_FIX_AST_APPEND
          si::appendStatement(block_gbuf, loop_body);
#else
          loop_body->append_statement(block_gbuf);
#endif
        }

        {
          // { long int j = s0.g->gridattr; 
          //   __PSSetKernelArg(argc, sizeof(j), &j); argc++; }
#ifdef DEBUG_FIX_AST_APPEND
          si::appendStatement(
            block_setkernelarg(
                argc_idx,
                sb::buildAssignInitializer(
                  sb::buildArrowExp(arg_sg, sb::buildVarRefExp("gridattr"))
            )),
            loop_body);
#else
          loop_body->append_statement(
            block_setkernelarg(
                argc_idx,
                sb::buildAssignInitializer(
                  sb::buildArrowExp(arg_sg, sb::buildVarRefExp("gridattr"))
                )));
#endif
        } //

        // skip index
        member++;
 
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
#ifdef DEBUG_FIX_AST_APPEND
      si::appendStatement(sb::buildExprStatement(exp_runkernel), loop_body);
#else
      loop_body->append_statement(sb::buildExprStatement(exp_runkernel));
#endif
    }

    // Swap not used
#if 0
#ifdef DEBUG_FIX_DUP_SGEXP
    stencil_var = sb::buildVarRefExp(stencil_name);
#endif
    appendGridSwap(sm, stencil_var, loop_body);
#endif

    // reset argc: argc = 0;
    SgExpression *argc_zero_lhs = sb::buildVarRefExp(argc_idx);
    SgExpression *argc_zero_rhs = sb::buildIntVal(0);
    SgExpression *argc_zero = sb::buildAssignOp(argc_zero_lhs, argc_zero_rhs);
#ifdef DEBUG_FIX_AST_APPEND
    si::appendStatement(sb::buildExprStatement(argc_zero), loop_body);
#else
    loop_body->append_statement(sb::buildExprStatement(argc_zero));
#endif

  } // NUMERATE(stencil_idx, it, run->stencils().begin(), run->stencils().end())

  return loop_body;

} // GenerateRunLoopBody



} // namespace translator
} // namespace physis
