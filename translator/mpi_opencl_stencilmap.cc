#include "translator/mpi_opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/mpi_opencl_runtime_builder.h"
#include "translator/reference_runtime_builder.h"
#include "translator/SageBuilderEx.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/mpi_opencl_optimizer.h"
#include "translator/rose_ast_attribute.h"

namespace pu = physis::util;
namespace sb = SageBuilder;
namespace si = SageInterface;
namespace sbx = physis::translator::SageBuilderEx;

namespace physis {
namespace translator {

SgVariableDeclaration *MPIOpenCLTranslator::BuildGridSizeDeclaration(
    const SgName &name,
    SgExpression *dom_dim_x,
    SgExpression *dom_dim_y,    
    SgExpression *block_dim_x,
    SgExpression *block_dim_y,
    SgScopeStatement *scope) {

  SgVariableDeclaration *sg_gb;
  // TODO: use ivec_type_ ?
  // size_t name_var[]
  SgType *type_sizet =
      sb::buildOpaqueType("size_t", scope);
  SgType *type_sg_gb =
      sb::buildArrayType(type_sizet);

  // Create array
  SgExprListExp *def_gb = sb::buildExprListExp();

  SgExpression *dim_x =
      sb::buildDivideOp(
          dom_dim_x,
          sb::buildCastExp(
              block_dim_x,
              sb::buildDoubleType()
                           )
                        );
  dim_x = BuildFunctionCall("ceil", dim_x);
  dim_x = sb::buildMultiplyOp(
      dim_x,
      block_dim_x
                              );
  dim_x = sb::buildCastExp(dim_x, type_sizet);

  SgExpression *dim_y =
      sb::buildDivideOp(
          dom_dim_y,
          sb::buildCastExp(
              block_dim_y,
              sb::buildDoubleType()
                           )
                        );
  dim_y = BuildFunctionCall("ceil", dim_y);
  dim_y = sb::buildMultiplyOp(
      dim_y,
      block_dim_y
                              );
  dim_y = sb::buildCastExp(dim_y, type_sizet);

  SgExpression *dim_z = sb::buildIntVal(1);

  si::appendExpression(def_gb, dim_x);
  si::appendExpression(def_gb, dim_y);
  si::appendExpression(def_gb, dim_z);

  SgAggregateInitializer *initval_gb =
      sb::buildAggregateInitializer(def_gb, type_sg_gb);

  // size_t name[] = {initval_gb};
  sg_gb = sb::buildVariableDeclaration(
      name, type_sg_gb, initval_gb, scope);

  // !! Note
  // This function calls "ceil", math.h needed
  LOG_INFO() << "Inserting <math.h> because of ceil call.\n";
  // TODO: The following does not work?
  // si::insertHeader("math.h", PreprocessingInfo::before, src_->get_globalScope());
  static int math_h_included_p = false;
  if (!math_h_included_p){
    std::string str_insert = "";
    str_insert += "#ifndef "; str_insert += kernel_mode_macro();
    si::attachArbitraryText(src_->get_globalScope(), str_insert);
    si::attachArbitraryText(src_->get_globalScope(), "#include <math.h>");
    si::attachArbitraryText(src_->get_globalScope(), "#endif");

    math_h_included_p = 1;
  }

  return sg_gb;
}


// REFACTORING
void MPIOpenCLTranslator::ProcessStencilMap(
    StencilMap *smap,
    SgVarRefExp *stencils,
    int stencil_map_index,
    Run *run,
    SgScopeStatement *function_body,
    SgScopeStatement *loop_body,
    SgVariableDeclaration *dec_local_size,
    SgVariableDeclaration *argc_idx
                                            ){

  // s0 and so on
  string stencil_name = "s" + toString(stencil_map_index);
  SgExpression *idx = sb::buildIntVal(stencil_map_index);
  // struct __PSStencilRun_kernel * and so on
  SgType *stencil_ptr_type = sb::buildPointerType(smap->stencil_type());
  // struct __PSStencil_kernel *s0 = (struct __PSStencil_kernel *)stencils[0];
  SgAssignInitializer *init =
      sb::buildAssignInitializer(
          sb::buildCastExp(
              sb::buildPntrArrRefExp(stencils, idx),
              stencil_ptr_type), stencil_ptr_type);
  SgVariableDeclaration *sdecl
      = sb::buildVariableDeclaration(stencil_name, stencil_ptr_type,
                                     init, function_body);
  SgVarRefExp *stencil_var = sb::buildVarRefExp(sdecl);
  si::appendStatement(sdecl, function_body);

  SgInitializedNamePtrList remote_grids;
  SgStatementPtrList load_statements;

  bool overlap_eligible;
  int overlap_width;
  GenerateLoadRemoteGridRegion(smap, stencil_var, run, loop_body,
                               remote_grids, load_statements,
                               overlap_eligible, overlap_width);
  bool overlap_enabled = flag_mpi_overlap_ &&  overlap_eligible;
  if (overlap_enabled) {
    LOG_INFO() << "Generating overlapping code\n";
  } else {
    LOG_INFO() << "Generating non-overlapping code\n";    
  }
  // run kernel function

  
  // Call the stencil kernel
  // Build an argument list by expanding members of the parameter struct
  // i.e. struct {a, b, c}; -> (s.a, s.b, s.c)

  SgClassDefinition *stencil_def = smap->GetStencilTypeDefinition();
  PSAssert(stencil_def);

  SgVariableDeclaration *grid_dim = BuildGridSizeDeclaration(
#if 0
      stencil_name + "_grid_dim",
#else
      "__PS_CL_globalsize_" + stencil_name,
#endif
      BuildGetLocalSize(sb::buildIntVal(0)),
      BuildGetLocalSize(sb::buildIntVal(1)),
      opencl_trans_->BuildBlockDimX(),
      opencl_trans_->BuildBlockDimY(),
      function_body);

  si::appendStatement(grid_dim, function_body);

  // Create three blocks
  SgBasicBlock *block_args_normal = sb::buildBasicBlock();
  SgBasicBlock *block_args_boundary = sb::buildBasicBlock();
  SgBasicBlock *block_args_boundary_multi = sb::buildBasicBlock();

  // Dom statement, needed later
  SgExpression *exp_dom = 0;

  // Prepare argc = 0;
  SgExpression *argc_zero;
  {
    SgExpression *argc_zero_lhs = sb::buildVarRefExp(argc_idx);
    SgExpression *argc_zero_rhs = sb::buildUnsignedIntVal(0);
    argc_zero = sb::buildAssignOp(argc_zero_lhs, argc_zero_rhs);
  }
  
  // Append the local offset
  // __PSGetLocalOffset(i)
  for (int i = 0; i < smap->getNumDim()-1; ++i) {
    SgBasicBlock *block_tmp = 
        opencl_trans_->block_setkernelarg(
            argc_idx,
            sb::buildAssignInitializer(
                BuildGetLocalOffset(sb::buildIntVal(i))
                                       ));
    si::appendStatement(
        si::copyStatement(block_tmp),
        block_args_normal);
    if (!flag_multistream_boundary_)
      si::appendStatement(
          si::copyStatement(block_tmp),
          block_args_boundary);

  } // for (int i = 0; i < smap->getNumDim()-1; ++i)

  if (!flag_multistream_boundary_) {
    SgBasicBlock *block_tmp = 
        opencl_trans_->block_setkernelarg(
            argc_idx,
            sb::buildAssignInitializer(
                sb::buildIntVal(overlap_width)
                                       ));
    si::appendStatement(block_tmp, block_args_boundary);

  } // if (!flag_multistream_boundary_)

  // Enumerate members of parameter struct
  const SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    SgExpression *arg =
        sb::buildArrowExp(stencil_var, sb::buildVarRefExp(member_decl));

    //const SgInitializedNamePtrList &vars = member_decl->get_variables();
    //GridType *gt = tx_->findGridType(vars[0]->get_type());
    SgType *member_type = si::getFirstVarType(member_decl);

    SgStatement *st_add = 0;
    SgStatement *st_add_normal = 0;
    SgStatement *st_add_boundary = 0;
    SgStatement *st_add_boundary_multi = 0;

    GridType *gt = tx_->findGridType(member_type);
    if (gt) {
      // Grid Type
      // __PSSetKernelArg_Grid3DFloat(&argc, (__PSGrid3DFloatDev *)(__PSGridGetDev(s0 -> gaa)));
      SgExprListExp *args_gridset =
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(argc_idx)),
              sb::buildCastExp(
                  BuildGridGetDev(arg),
                  sb::buildPointerType(BuildOnDeviceGridType(gt, 0))
                               ),
              NULL
                               );

      SgType *ty = gt->getElmType();
      std::string callname = "__PSSetKernelArg_Grid3D";
      if (isSgTypeDouble(ty)) {
        callname += "Double";
      } else {
        callname += "Float";
      }

      SgFunctionCallExp *call_gridset = 
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp(callname), args_gridset);
      st_add = sb::buildExprStatement(call_gridset);
      st_add_normal = si::copyStatement(st_add);
      st_add_boundary = si::copyStatement(st_add);
      st_add_boundary_multi = si::copyStatement(st_add);

      // skip the grid index
      ++member;
    }

    if (Domain::isDomainType(member_type)) {
      // Domain type

      // Set dom type
      exp_dom = si::copyExpression(arg);

      // Create inner block
      SgBasicBlock *block_dom_arg = sb::buildBasicBlock();
      SgBasicBlock *block_dom_boundary = sb::buildBasicBlock();
      // __PSDomain dom_tmp = foo;
      SgVariableDeclaration *dec_dom_arg =
          sb::buildVariableDeclaration(
              "dom_tmp", member_type, NULL, block_dom_arg);
      SgVariableDeclaration *dec_dom_boundary =
          sb::buildVariableDeclaration(
              "dom_tmp", member_type, NULL, block_dom_boundary);

      // Default arg
      SgExpression *exp_init_arg = si::copyExpression(arg);
      SgExpression *exp_init_boundary = si::copyExpression(arg);

      if (overlap_enabled){
        // __PSDomainShrink(&s0->dom, overlap_width)
        exp_init_arg = BuildDomainShrink(
            sb::buildAddressOfOp(exp_init_arg), sb::buildIntVal(overlap_width)
                                         );
      }
      dec_dom_arg->reset_initializer(
          sb::buildAssignInitializer(
              exp_init_arg, member_type));
      dec_dom_boundary->reset_initializer(
          sb::buildAssignInitializer(
              exp_init_boundary, member_type));
      si::appendStatement(dec_dom_arg, block_dom_arg);
      si::appendStatement(dec_dom_boundary, block_dom_boundary);

      SgExprListExp *args_domset_arg =
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(argc_idx)),
              sb::buildAddressOfOp(sb::buildVarRefExp(dec_dom_arg)),
              NULL
                               );
      SgExprListExp *args_domset_boundary =
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(argc_idx)),
              sb::buildAddressOfOp(sb::buildVarRefExp(dec_dom_boundary)),
              NULL
                               );
      SgFunctionCallExp *call_domset_arg =
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSSetKernelArg_Dom"), args_domset_arg);
      SgFunctionCallExp *call_domset_boundary =
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSSetKernelArg_Dom"), args_domset_boundary);

      SgExprStatement *state_call_domset_arg = 
          sb::buildExprStatement(call_domset_arg);
      SgExprStatement *state_call_domset_boundary = 
          sb::buildExprStatement(call_domset_boundary);
      si::appendStatement(state_call_domset_arg, block_dom_arg);
      si::appendStatement(state_call_domset_boundary, block_dom_boundary);

      st_add = si::copyStatement(block_dom_arg);
      st_add_normal = si::copyStatement(block_dom_arg);
      st_add_boundary = si::copyStatement(block_dom_boundary);

    } // if (Domain::isDomainType(member_type))

    if (!st_add) { // normal case
      st_add = 
          opencl_trans_->block_setkernelarg(
              argc_idx,
              sb::buildAssignInitializer(arg),
              member_type
                                            );
      st_add_normal = si::copyStatement(st_add);
      st_add_boundary = si::copyStatement(st_add);
      st_add_boundary_multi = si::copyStatement(st_add);
    }

    if (st_add_normal)
      si::appendStatement(st_add_normal, block_args_normal);
    if (st_add_boundary)
      si::appendStatement(st_add_boundary, block_args_boundary);
    if (st_add_boundary_multi)
      si::appendStatement(st_add_boundary_multi, block_args_boundary_multi);

  } //  FOREACH(member, members.begin(), members.end())

  // Generate Kernel invocation code

  if (overlap_enabled) {
    SgVarRefExp *inner_stream = sb::buildVarRefExp("stream_inner");
    PSAssert(inner_stream);

    SgFunctionSymbol *fs_inner =
        rose_util::getFunctionSymbol(smap->run_inner());

    {
      SgExprListExp *arg_setkernel =
          sb::buildExprListExp(
              sb::buildStringVal(fs_inner->get_name().getString()),
              sb::buildVarRefExp("USE_INNER", block_args_normal),
              sb::buildIntVal(0),
              sb::buildIntVal(0),
              NULL
                               );
      SgFunctionCallExp *call_setkernel =
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSSetKernel"), arg_setkernel);
      // Not append but prepend!!
      si::prependStatement(sb::buildExprStatement(call_setkernel), block_args_normal);
    }
    {
      SgExprListExp *arg_runkernel =
          sb::buildExprListExp(
              sb::buildVarRefExp(grid_dim),
              sb::buildVarRefExp(dec_local_size),
              NULL
                               );
      SgFunctionCallExp *call_runkernel =
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSRunKernel"), arg_runkernel);
      si::appendStatement(sb::buildExprStatement(call_runkernel), block_args_normal);
    }
    // Reset argc
    si::appendStatement(sb::buildExprStatement(argc_zero), block_args_normal);
    si::appendStatement(block_args_normal, loop_body);

    // perform boundary exchange concurrently
    FOREACH (sit, load_statements.begin(), load_statements.end()) {
      si::appendStatement(*sit, loop_body);

    }
    LOG_INFO() << "generating call to boundary kernel\n";
    if (overlap_width && !flag_multistream_boundary_) {

      LOG_INFO() << "single-stream version\n";
      {
        SgExprListExp *arg_setkernel =
            sb::buildExprListExp(
                sb::buildStringVal(smap->getRunName() + GetBoundarySuffix()),
                sb::buildVarRefExp("USE_GENERIC", block_args_boundary),
                sb::buildIntVal(0),
                sb::buildIntVal(0),
                NULL
                                 );
        SgFunctionCallExp *call_setkernel =
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("__PSSetKernel"), arg_setkernel);
        // Not append but prepend!!
        si::prependStatement(sb::buildExprStatement(call_setkernel), block_args_boundary);
      }
      {
        SgExprListExp *arg_runkernel =
            sb::buildExprListExp(
                sb::buildVarRefExp(grid_dim),
                sb::buildVarRefExp(dec_local_size),
                NULL
                                 );
        SgFunctionCallExp *call_runkernel =
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("__PSRunKernel"), arg_runkernel);
        si::appendStatement(sb::buildExprStatement(call_runkernel), block_args_boundary);
      }
      // Reset argc
      si::appendStatement(sb::buildExprStatement(argc_zero), block_args_boundary);
      si::appendStatement(block_args_boundary, loop_body);

    } else if (overlap_width) {
      LOG_INFO() << "multi-stream version\n";
      // rose_util::AppendExprStatement(
      //     loop_body,
      //     BuildCudaStreamSynchronize(sb::buildVarRefExp("stream_boundary_copy")));

      SgExpression *dom = exp_dom;
      PSAssert(dom);     

      // 6 streams for
      int stream_index = 0;
      int num_x_streams = 5;
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < num_x_streams; ++i) {

          SgBasicBlock *block_boundary_multi = isSgBasicBlock(
              si::copyStatement(block_args_boundary_multi));
          SgExpression *bd =
              BuildDomainGetBoundary(sb::buildAddressOfOp(dom),
                                     0, j, sb::buildIntVal(overlap_width),
                                     num_x_streams, i);
          {
            SgBasicBlock *block_tmp = sb::buildBasicBlock();
            SgVariableDeclaration *dec_domset = 
                sb::buildVariableDeclaration(
                    "dom_tmp", dom_type_,
                    sb::buildAssignInitializer(bd), block_tmp);
            si::appendStatement(dec_domset, block_tmp);
            {
              SgExprListExp *args_domset=
                  sb::buildExprListExp(
                      sb::buildAddressOfOp(sb::buildVarRefExp(argc_idx)),
                      sb::buildAddressOfOp(sb::buildVarRefExp(dec_domset)),
                      NULL
                                       );
              SgFunctionCallExp *call_domset =
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("__PSSetKernelArg_Dom"), args_domset);
              si::appendStatement(sb::buildExprStatement(call_domset), block_tmp);
            }
 
            // Not append but prepend !!
            si::prependStatement(block_tmp, block_boundary_multi);
          }
          {
            int dimz = 512 / (overlap_width * 128);

            SgType *type_sizet = sb::buildOpaqueType("size_t", global_scope_);
            SgType *type_sg_gb = sb::buildArrayType(type_sizet);
            SgAggregateInitializer *initval_gb =
                sb::buildAggregateInitializer(
                    sb::buildExprListExp(
                        sb::buildIntVal(overlap_width),
                        sb::buildIntVal(128),
                        sb::buildIntVal(dimz),
                        NULL
                                         ),
                    type_sg_gb
                                              );

            SgVariableDeclaration *dec_local_size_tmp =
                sb::buildVariableDeclaration(
                    "local_size_tmp", type_sg_gb, initval_gb, block_boundary_multi);
            SgVariableDeclaration *dec_global_size_tmp =
                sb::buildVariableDeclaration(
                    "global_size_tmp", type_sg_gb, initval_gb, block_boundary_multi);
            si::appendStatement(dec_global_size_tmp, block_boundary_multi);
            si::appendStatement(dec_local_size_tmp, block_boundary_multi);
            {
              SgExprListExp *arg_setkernel =
                  sb::buildExprListExp(
                      sb::buildStringVal(smap->getRunName() + GetBoundarySuffix(0, j)),
                      sb::buildVarRefExp("USE_BOUNDARY_KERNEL", block_boundary_multi),
                      sb::buildIntVal(stream_index),
                      sb::buildIntVal(0),
                      NULL
                                       );
              ++stream_index;
              SgFunctionCallExp *call_setkernel =
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("__PSSetKernel"), arg_setkernel);
              // Not append but prepend!!
              si::prependStatement(sb::buildExprStatement(call_setkernel), block_boundary_multi);
            }

            {
              SgExprListExp *arg_runkernel =
                  sb::buildExprListExp(
                      sb::buildVarRefExp(dec_global_size_tmp),
                      sb::buildVarRefExp(dec_local_size_tmp),
                      NULL
                                       );
              SgFunctionCallExp *call_runkernel =
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("__PSRunKernel"), arg_runkernel);
              si::appendStatement(sb::buildExprStatement(call_runkernel), block_boundary_multi);
            }

          }
          // reset argc
          si::appendStatement(sb::buildExprStatement(argc_zero), block_boundary_multi);
          si::appendStatement(block_boundary_multi, loop_body);
        }
      }


      for (int j = 1; j < 3; ++j) {
        for (int i = 0; i < 2; ++i) {
          SgBasicBlock *block_boundary_multi = isSgBasicBlock(
              si::copyStatement(block_args_boundary_multi));
          SgExpression *bd =
              BuildDomainGetBoundary(sb::buildAddressOfOp(dom),
                                     j, i, sb::buildIntVal(overlap_width),
                                     1, 0);
          {
            SgBasicBlock *block_tmp = sb::buildBasicBlock();
            SgVariableDeclaration *dec_domset = 
                sb::buildVariableDeclaration(
                    "dom_tmp", dom_type_,
                    sb::buildAssignInitializer(bd), block_tmp);
            si::appendStatement(dec_domset, block_tmp);
            {
              SgExprListExp *args_domset=
                  sb::buildExprListExp(
                      sb::buildAddressOfOp(sb::buildVarRefExp(argc_idx)),
                      sb::buildAddressOfOp(sb::buildVarRefExp(dec_domset)),
                      NULL
                                       );
              SgFunctionCallExp *call_domset =
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("__PSSetKernelArg_Dom"), args_domset);
              si::appendStatement(sb::buildExprStatement(call_domset), block_tmp);
            }
 
            // Not append but prepend !!
            si::prependStatement(block_tmp, block_boundary_multi);
          }
          {
            int dimz = 512 / (overlap_width * 128);

            SgType *type_sizet = sb::buildOpaqueType("size_t", global_scope_);
            SgType *type_sg_gb = sb::buildArrayType(type_sizet);
            SgAggregateInitializer *initval_gb;
            SgAggregateInitializer *initval_local;
            if (j == 1) {
              initval_gb = sb::buildAggregateInitializer(
                  sb::buildExprListExp(
                      sb::buildIntVal(128),
                      sb::buildIntVal(overlap_width),
                      sb::buildIntVal(dimz),
                      NULL
                                       ),
                  type_sg_gb
                                                         );
              initval_local = sb::buildAggregateInitializer(
                  sb::buildExprListExp(
                      sb::buildIntVal(64),
                      sb::buildIntVal(overlap_width),
                      sb::buildIntVal(dimz),
                      NULL
                                       ),
                  type_sg_gb
                                                            );
            } else {
              initval_gb = sb::buildAggregateInitializer(
                  sb::buildExprListExp(
                      sb::buildIntVal(128),
                      sb::buildIntVal(4),
                      sb::buildIntVal(1),
                      NULL
                                       ),
                  type_sg_gb
                                                         );
              initval_local = sb::buildAggregateInitializer(
                  sb::buildExprListExp(
                      sb::buildIntVal(64),
                      sb::buildIntVal(4),
                      sb::buildIntVal(1),
                      NULL
                                       ),
                  type_sg_gb
                                                            );
            }

            SgVariableDeclaration *dec_local_size_tmp =
                sb::buildVariableDeclaration(
                    "local_size_tmp", type_sg_gb, initval_local, block_boundary_multi);
            SgVariableDeclaration *dec_global_size_tmp =
                sb::buildVariableDeclaration(
                    "global_size_tmp", type_sg_gb, initval_gb, block_boundary_multi);
            si::appendStatement(dec_global_size_tmp, block_boundary_multi);
            si::appendStatement(dec_local_size_tmp, block_boundary_multi);
            {
              SgExprListExp *arg_setkernel =
                  sb::buildExprListExp(
                      sb::buildStringVal(smap->getRunName() + GetBoundarySuffix(j, i)),
                      sb::buildVarRefExp("USE_BOUNDARY_KERNEL", block_boundary_multi),
                      sb::buildIntVal(stream_index),
                      sb::buildIntVal(0),
                      NULL
                                       );
              ++stream_index;
              SgFunctionCallExp *call_setkernel =
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("__PSSetKernel"), arg_setkernel);
              // Not append but prepend!!
              si::prependStatement(sb::buildExprStatement(call_setkernel), block_boundary_multi);
            }

            {
              SgExprListExp *arg_runkernel =
                  sb::buildExprListExp(
                      sb::buildVarRefExp(dec_global_size_tmp),
                      sb::buildVarRefExp(dec_local_size_tmp),
                      NULL
                                       );
              SgFunctionCallExp *call_runkernel =
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("__PSRunKernel"), arg_runkernel);
              si::appendStatement(sb::buildExprStatement(call_runkernel), block_boundary_multi);
            }
          }
          // reset argc
          si::appendStatement(sb::buildExprStatement(argc_zero), block_boundary_multi);
          si::appendStatement(block_boundary_multi, loop_body);
        }
      }
      si::appendStatement(
          sb::buildExprStatement(BuildCLThreadSynchronize()),
          loop_body);
    }

  } else {
    // perform boundary exchange before kernel invocation synchronously
    FOREACH (sit, load_statements.begin(), load_statements.end()) {
      loop_body->append_statement(*sit);
    }
    {
      SgExprListExp *arg_setkernel =
          sb::buildExprListExp(
              sb::buildStringVal(smap->getRunName()),
              sb::buildVarRefExp("USE_GENERIC", block_args_normal),
              sb::buildIntVal(0),
              sb::buildIntVal(0),
              NULL
                               );
      SgFunctionCallExp *call_setkernel =
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSSetKernel"), arg_setkernel);
      // Not append but prepend!!
      si::prependStatement(sb::buildExprStatement(call_setkernel), block_args_normal);
    }
    {
      SgExprListExp *arg_runkernel =
          sb::buildExprListExp(
              sb::buildVarRefExp(grid_dim),
              sb::buildVarRefExp(dec_local_size),
              NULL
                               );
      SgFunctionCallExp *call_runkernel =
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("__PSRunKernel"), arg_runkernel);
      si::appendStatement(sb::buildExprStatement(call_runkernel), block_args_normal);
    }
    // Reset argc
    si::appendStatement(sb::buildExprStatement(argc_zero), block_args_normal);
    si::appendStatement(block_args_normal, loop_body);
  }
  appendGridSwap(smap, stencil_var, loop_body);
  DeactivateRemoteGrids(smap, stencil_var, loop_body,
                        remote_grids);

  FixGridAddresses(smap, stencil_var, function_body);
}


} // namespace translator
} // namespace physis
