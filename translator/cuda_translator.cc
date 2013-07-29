// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/cuda_translator.h"

#include <algorithm>
#include <string>

#include "translator/cuda_runtime_builder.h"
#include "translator/SageBuilderEx.h"
#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/cuda_builder.h"
#include "translator/stencil_analysis.h"
#include "translator/physis_names.h"

namespace pu = physis::util;
namespace sb = SageBuilder;
namespace si = SageInterface;
namespace sbx = physis::translator::SageBuilderEx;

#define BLOCK_DIM_X_DEFAULT (64)
#define BLOCK_DIM_Y_DEFAULT (4)
#define BLOCK_DIM_Z_DEFAULT (1)

namespace physis {
namespace translator {

CUDATranslator::CUDATranslator(const Configuration &config):
    ReferenceTranslator(config),
    block_dim_x_(BLOCK_DIM_X_DEFAULT),
    block_dim_y_(BLOCK_DIM_Y_DEFAULT),
    block_dim_z_(BLOCK_DIM_Z_DEFAULT) {
  target_specific_macro_ = "PHYSIS_CUDA";
  //validate_ast_ = false;
  validate_ast_ = true;  
  // Redefine the block size if specified in the configuration file
  cuda_block_size_vals_.clear();  /* clear */
  const pu::LuaValue *lv = config.Lookup(Configuration::CUDA_BLOCK_SIZE);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    const pu::LuaTable *tbl2 = tbl->lst().begin()->second->getAsLuaTable();
    if (tbl2) {
      if (tbl->lst().size() == 1 || !config.auto_tuning()) {
        tbl = tbl2; /* use first one */
      } else {
        block_dim_x_ = block_dim_y_ = block_dim_z_ = 0;
        /* get all selection from CUDA_BLOCK_SIZE */
        FOREACH (it, tbl->lst().begin(), tbl->lst().end()) {
          std::vector<SgExpression *> iv;
          std::vector<double> v;
          PSAssert(tbl2 = it->second->getAsLuaTable());
          PSAssert(tbl2->get(v));
          iv.push_back(sb::buildIntVal((int)v[0]));
          iv.push_back(sb::buildIntVal((int)v[1]));
          iv.push_back(sb::buildIntVal((int)v[2]));
          cuda_block_size_vals_.push_back(sb::buildAggregateInitializer(sb::buildExprListExp(iv)));
        }
        return;
      }
    }
    std::vector<double> v;
    PSAssert(tbl->get(v));
    block_dim_x_ = (int)v[0];
    block_dim_y_ = (int)v[1];
    block_dim_z_ = (int)v[2];
  }
}

void CUDATranslator::SetUp(SgProject *project,
                           TranslationContext *context,
                           RuntimeBuilder *rt_builder) {
  ReferenceTranslator::SetUp(project, context, rt_builder);

  /* auto tuning & has dynamic arguments */
  if (!(config_.auto_tuning() && config_.ndynamic() > 1)) return;
  /* build __cuda_block_size_struct */
  SgClassDeclaration *s =
      sb::buildStructDeclaration(
          SgName("__cuda_block_size_struct"), global_scope_);
  si::appendStatement(
      sb::buildVariableDeclaration("x", sb::buildIntType()),
      s->get_definition());
  si::appendStatement(
      sb::buildVariableDeclaration("y", sb::buildIntType()),
      s->get_definition());
  si::appendStatement(
      sb::buildVariableDeclaration("z", sb::buildIntType()),
      s->get_definition());
  cuda_block_size_type_ = s->get_type();
  SgVariableDeclaration *cuda_block_size =
      sb::buildVariableDeclaration(
          "__cuda_block_size",
          sb::buildConstType(
              sb::buildArrayType(cuda_block_size_type_)),
          sb::buildAggregateInitializer(
              sb::buildExprListExp(cuda_block_size_vals_)),
          global_scope_);
  si::setStatic(cuda_block_size);
  si::prependStatement(cuda_block_size, si::getFirstGlobalScope(project_));
  si::prependStatement(s, si::getFirstGlobalScope(project_));
}


void CUDATranslator::appendNewArgExtra(SgExprListExp *args,
                                       Grid *g,
                                       SgVariableDeclaration *dim_decl) {
  GridType *gt = g->getType();
  SgExpression *extra_arg = NULL;
  if (gt->IsPrimitivePointType()) {
    extra_arg = rose_util::buildNULL(global_scope_);
  } else {
    extra_arg = sb::buildFunctionRefExp(gt->aux_new_decl());
  }
  si::appendExpression(args, extra_arg);
  return;
}

void CUDATranslator::TranslateKernelDeclaration(
    SgFunctionDeclaration *node) {
  SgFunctionModifier &modifier = node->get_functionModifier();
  modifier.setCudaDevice();

  // e.g., PSGrid3DFloat -> __PSGrid3DFloatDev *
  Rose_STL_Container<SgNode*> exps =
      NodeQuery::querySubTree(node, V_SgInitializedName);
  FOREACH (it, exps.begin(), exps.end()) {
    SgInitializedName *exp = isSgInitializedName(*it);
    PSAssert(exp);
    SgType *cur_type = exp->get_type();
    GridType *gt = tx_->findGridType(cur_type);
    // not a grid type
    if (!gt) continue;
    SgType *new_type = sb::buildPointerType(BuildOnDeviceGridType(gt));
    exp->set_type(new_type);
  }
  return;
}

void CUDATranslator::Visit(SgExpression *node) {
  // Replace get(g, offset).x with g->x[offset]
  if (isSgDotExp(node)) {
    SgDotExp *dot = isSgDotExp(node);
    SgExpression *lhs = dot->get_lhs_operand();
    GridGetAttribute *gga =
        rose_util::GetASTAttribute<GridGetAttribute>(lhs);
    if (gga == NULL) return;
    TranslateGetForUserDefinedType(dot, NULL);
  }
  SgPntrArrRefExp *par = isSgPntrArrRefExp(node);  
  // Process the top-level array access expression
  if (par == NULL ||
      isSgPntrArrRefExp(node->get_parent())) return;  

  SgExpression *lhs = NULL;  
  while (true) {
    lhs = par->get_lhs_operand();
    if (isSgPntrArrRefExp(lhs)) {
      par = isSgPntrArrRefExp(lhs);
      continue;
    } else {
      break;
    }
  }
  if (isSgDotExp(lhs)) {
    SgDotExp *dot = isSgDotExp(lhs);
    SgExpression *dot_lhs = dot->get_lhs_operand();
    GridGetAttribute *gga =
        rose_util::GetASTAttribute<GridGetAttribute>(dot_lhs);
    if (gga == NULL) return;
    TranslateGetForUserDefinedType(dot, isSgPntrArrRefExp(node));
    return;
  }
  
  return;
}

void CUDATranslator::TranslateGetForUserDefinedType(
    SgDotExp *node, SgPntrArrRefExp *array_top) {
  SgVarRefExp *mem_ref =
      isSgVarRefExp(node->get_rhs_operand());
  PSAssert(mem_ref);
  if (isSgPntrArrRefExp(node->get_parent()) &&
      array_top == NULL) {
    return;
  }
  SgExpression *get_exp = node->get_lhs_operand();
  GridGetAttribute *gga =
      rose_util::GetASTAttribute<GridGetAttribute>(
          get_exp);
  const string &mem_name = rose_util::GetName(mem_ref);
  SgExpressionPtrList indices =
      GridOffsetAnalysis::GetIndices(
          GridGetAnalysis::GetOffset(get_exp));
  SgExpressionPtrList args;
  rose_util::CopyExpressionPtrList(indices, args);
  SgInitializedName *gv = GridGetAnalysis::GetGridVar(get_exp);
  GridType *gt = rose_util::GetASTAttribute<GridType>(gv);
  SgExpression *original = node;
  SgExpression *new_get = NULL;
  if (array_top == NULL) {
    new_get = 
        rt_builder_->BuildGridGet(
            sb::buildVarRefExp(gv->get_name(),
                               si::getScope(node)),
            rose_util::GetASTAttribute<GridVarAttribute>(gv),
            gt, &args,
            gga->GetStencilIndexList(),
            gga->in_kernel(), gga->is_periodic(),
            mem_name);
  } else {
    // Member is an array    
    SgExpressionVector indices;
    SgExpression *parent;
    PSAssert(AnalyzeGetArrayMember(node, indices, parent));
    rose_util::ReplaceWithCopy(indices);
    PSAssert(array_top == parent);
    original = parent;
    new_get = 
        rt_builder_->BuildGridGet(
            sb::buildVarRefExp(gv->get_name(),
                               si::getScope(node)),
            rose_util::GetASTAttribute<GridVarAttribute>(gv),            
            gt, &args,
            gga->GetStencilIndexList(),
            gga->in_kernel(), gga->is_periodic(),
            mem_name,
            indices);
  }
  // Replace the parent expression
  si::replaceExpression(original, new_get);
  return;  
}

void CUDATranslator::TranslateGet(SgFunctionCallExp *func_call_exp,
                                  SgInitializedName *grid_arg,
                                  bool is_kernel, bool is_periodic) {
  GridType *gt = tx_->findGridType(grid_arg->get_type());
  SgExpressionPtrList args;
  rose_util::CopyExpressionPtrList(
      func_call_exp->get_args()->get_expressions(), args);
  const StencilIndexList *sil =
      rose_util::GetASTAttribute<GridGetAttribute>(
          func_call_exp)->GetStencilIndexList();
  SgExpression *gv = sb::buildVarRefExp(grid_arg->get_name(),
                                        si::getScope(func_call_exp));
  SgExpression *real_get =
      rt_builder_->BuildGridGet(
          gv,
          rose_util::GetASTAttribute<GridVarAttribute>(grid_arg),
          gt, &args, sil, is_kernel, is_periodic);
  si::replaceExpression(func_call_exp, real_get);
}

void CUDATranslator::TranslateEmit(SgFunctionCallExp *node,
                                   GridEmitAttribute *attr) {
  SgInitializedName *gv = attr->gv();  
  bool is_grid_type_specific_call =
      GridType::isGridTypeSpecificCall(node);

  GridType *gt = tx_->findGridType(gv->get_type());
  if (gt->IsPrimitivePointType()) {
    ReferenceTranslator::TranslateEmit(node, attr);
    return;
  }

  // build a function call to gt->aux_emit_decl(g, offset, v)
  int nd = gt->getNumDim();
  SgExpressionPtrList args;
  SgInitializedNamePtrList &params =
      getContainingFunction(node)->get_args();  
  for (int i = 0; i < nd; ++i) {
    SgInitializedName *p = params[i];
    args.push_back(sb::buildVarRefExp(p));
  }

  SgExpression *v =
      si::copyExpression(node->get_args()->get_expressions().back());
  
  SgExpression *real_exp = 
      rt_builder_->BuildGridEmit(
          sb::buildVarRefExp(attr->gv()), attr, &args, v, si::getScope(node));
  
  si::replaceExpression(node, real_exp);

  if (!is_grid_type_specific_call) {
    RemoveEmitDummyExp(real_exp);
  }
  
}

SgVariableDeclaration *CUDATranslator::BuildGridDimDeclaration(
    const SgName &name,
    int dim,
    SgExpression *dom_dim_x,
    SgExpression *dom_dim_y,    
    SgExpression *block_dim_x,
    SgExpression *block_dim_y,
    SgScopeStatement *scope) const {
  SgExpression *dim_x =
      sb::buildDivideOp(dom_dim_x,
                        sb::buildCastExp(block_dim_x,
                                         sb::buildDoubleType()));
  dim_x = BuildFunctionCall("ceil", dim_x);
  dim_x = sb::buildCastExp(dim_x, sb::buildIntType());
  SgExpression *dim_y = NULL;  
  if (dim >= 2) {
    dim_y =
      sb::buildDivideOp(dom_dim_y,
                        sb::buildCastExp(block_dim_y,
                                         sb::buildDoubleType()));
    dim_y = BuildFunctionCall("ceil", dim_y);
    dim_y = sb::buildCastExp(dim_y, sb::buildIntType());
  } else {
    dim_y = sb::buildIntVal(1);
  }
  SgExpression *dim_z = sb::buildIntVal(1);
  SgVariableDeclaration *grid_dim =
      sbx::buildDim3Declaration(name, dim_x, dim_y, dim_z, scope);
  return grid_dim;
}


SgExpression *CUDATranslator::BuildBlockDimX(int nd) {
  if (block_dim_x_ <= 0) {
    /* auto tuning & has dynamic arguments */
    return sb::buildVarRefExp("x");
  }
  return sb::buildIntVal(block_dim_x_);
}

SgExpression *CUDATranslator::BuildBlockDimY(int nd) {
  if (nd < 2) {
    return sb::buildIntVal(1);
  }
  if (block_dim_y_ <= 0) {
    /* auto tuning & has dynamic arguments */
    return sb::buildVarRefExp("y");
  }
  return sb::buildIntVal(block_dim_y_);  
}

SgExpression *CUDATranslator::BuildBlockDimZ(int nd) {
  if (nd < 3) {
    return sb::buildIntVal(1);
  }
  if (block_dim_z_ <= 0) {
    /* auto tuning & has dynamic arguments */
    return sb::buildVarRefExp("z");
  }
  return sb::buildIntVal(block_dim_z_);
}

void CUDATranslator::BuildRunBody(
    SgBasicBlock *block, Run *run, SgFunctionDeclaration *run_func) {
  // int i;
  SgVariableDeclaration *loop_index =
      sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  si::appendStatement(loop_index, block);
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
  SgBasicBlock *loop_body = BuildRunLoopBody(run, block);
  SgForStatement *loop =
      sb::buildForStatement(loop_init, loop_test, loop_incr, loop_body);

  TraceStencilRun(run, loop, block);
  
  // cudaThreadSynchronize after each loop if requested
  if (config_.LookupFlag(Configuration::CUDA_KERNEL_ERROR_CHECK)) {
    si::insertStatementAfter(
        loop,
        sb::buildExprStatement(BuildCudaDeviceSynchronize()));
    si::insertStatementBefore(
        loop,
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("cudaGetLastError"), NULL)));
    si::insertStatementAfter(
        loop,
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("__PSCheckCudaError"),
                sb::buildExprListExp(
                    sb::buildStringVal("Kernel Execution Failed!")))));
  }
  
  return;
}

SgExprListExp *CUDATranslator::BuildCUDAKernelArgList(
    int stencil_idx, StencilMap *sm, const string &sv_name) const {
  // Build an argument list by expanding members of the parameter struct
  // e.g., struct {a, b, c}; -> (s.a, s.b, s.c)
  SgExprListExp *args = sb::buildExprListExp();
  SgClassDefinition *stencil_def = sm->GetStencilTypeDefinition();
  PSAssert(stencil_def);

  // Enumerate members of parameter struct
  const SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    SgExpression *arg =
        sb::buildDotExp(sb::buildVarRefExp(sv_name),
                        sb::buildVarRefExp(member_decl));
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    // If the type of the member is grid, pass the device pointer.
    GridType *gt = tx_->findGridType(vars[0]->get_type());
    if (gt) {
      arg = sb::buildPointerDerefExp(
          sb::buildCastExp(
              sb::buildArrowExp(arg, sb::buildVarRefExp("dev")),
              sb::buildPointerType(BuildOnDeviceGridType(gt))));
      // skip the grid index
      ++member;
    }
    si::appendExpression(args, arg);
  }
  return args;
}

SgBasicBlock *CUDATranslator::BuildRunLoopBody(
    Run *run, SgScopeStatement *outer_block) {
  
  SgBasicBlock *loop_body = sb::buildBasicBlock();

  // Generates a call to each of the stencil function specified in the
  // PSStencilRun.
  ENUMERATE(stencil_idx, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *sm = it->second;
    int nd = sm->getNumDim();

    // Generate cache config code for each kernel
    SgFunctionSymbol *func_sym = rose_util::getFunctionSymbol(sm->run());
    PSAssert(func_sym);
#if 0 
    // This is done device-wide at the start-up time
    // Set the SM on-chip memory to prefer the L1 cache
    SgFunctionCallExp *cache_config =
        sbx::buildCudaCallFuncSetCacheConfig(func_sym,
                                             sbx::cudaFuncCachePreferL1,
                                             global_scope_);
    // Append invocation statement ahead of the loop
    si::appendStatement(sb::buildExprStatement(cache_config), outer_block);
#endif    

    string stencil_name = "s" + toString(stencil_idx);
    SgExprListExp *args = BuildCUDAKernelArgList(
        stencil_idx, sm, stencil_name);

    SgVariableDeclaration *block_dim =
        sbx::buildDim3Declaration(
            stencil_name + "_block_dim",
            BuildBlockDimX(nd),
            BuildBlockDimY(nd),
            BuildBlockDimZ(nd),
            outer_block);
    si::appendStatement(block_dim, outer_block);

    SgExpression *dom_max0 =
        sb::buildPntrArrRefExp(
            sb::buildDotExp(
                sb::buildDotExp(sb::buildVarRefExp(stencil_name),
                                sb::buildVarRefExp(GetStencilDomName())),
            sb::buildVarRefExp("local_max")),
        sb::buildIntVal(0));
    SgExpression *dom_max1 =
        sb::buildPntrArrRefExp(
            sb::buildDotExp(
                sb::buildDotExp(sb::buildVarRefExp(stencil_name),
                                sb::buildVarRefExp(GetStencilDomName())),
            sb::buildVarRefExp("local_max")),
        sb::buildIntVal(1));

    if (sm->IsRedBlackVariant()) {
      dom_max0 = sb::buildDivideOp(dom_max0, sb::buildIntVal(2));
    }

    SgVariableDeclaration *grid_dim =
        BuildGridDimDeclaration(stencil_name + "_grid_dim",
                                sm->getNumDim(),
                                dom_max0,
                                dom_max1,
                                BuildBlockDimX(nd),
                                BuildBlockDimY(nd),
                                outer_block);
    si::appendStatement(grid_dim, outer_block);

    // Generate Kernel invocation code
    SgCudaKernelExecConfig *cuda_config =
        sbx::buildCudaKernelExecConfig(sb::buildVarRefExp(grid_dim),
                                       sb::buildVarRefExp(block_dim),
                                       NULL, NULL);
    SgCudaKernelCallExp *cuda_call =
        sbx::buildCudaKernelCallExp(sb::buildFunctionRefExp(func_sym),
                                    args, cuda_config);
    si::appendStatement(sb::buildExprStatement(cuda_call), loop_body);    
    if (sm->IsRedBlackVariant()) {
      if (sm->IsRedBlack()) {
        SgCudaKernelCallExp *black_call =
            isSgCudaKernelCallExp(si::copyExpression(cuda_call));
        si::appendExpression(black_call->get_args(),
                             sb::buildIntVal(1));
        si::appendStatement(sb::buildExprStatement(black_call), loop_body);
      }
      si::appendExpression(
          cuda_call->get_args(),
          sb::buildIntVal(sm->IsBlack() ? 1 : 0));
    }
    //appendGridSwap(sm, stencil_name, false, loop_body);
  }
  return loop_body;
}

void CUDATranslator::ProcessUserDefinedPointType(
    SgClassDeclaration *grid_decl, GridType *gt) {
  LOG_DEBUG() << "Define grid data type for device.\n";
  SgClassDeclaration *type_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridDevTypeForUserType(grid_decl, gt);
  si::insertStatementAfter(grid_decl, type_decl);
  // If these user-defined types are defined in header files,
  // inserting new related types and declarations AFTER those types
  // does not seem to mean they are actually inserted after. New
  // declarations are actually inserted before the include
  // preprocessing directive, so the declaring header may not be included
  // at the time when the new declarations appear. Explicitly add new
  // include directive to work around this problem.
  if (!grid_decl->get_file_info()->isSameFile(src_)) {
    string fname = grid_decl->get_file_info()->get_filenameString();
    LOG_DEBUG() << "fname: " << fname << "\n";
    si::attachArbitraryText(type_decl, "#include \"" + fname + "\"\n",
                            PreprocessingInfo::before);
  }
  LOG_DEBUG() << "GridDevType: "
              << type_decl->unparseToString() << "\n";
  PSAssert(gt->aux_type() == NULL);
  gt->aux_type() = type_decl->get_type();
  gt->aux_decl() = type_decl;

  // Build GridNew for this type
  SgFunctionDeclaration *new_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridNewFuncForUserType(gt);
  si::insertStatementAfter(type_decl, new_decl);
  gt->aux_new_decl() = new_decl;
  // Build GridFree for this type
  SgFunctionDeclaration *free_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridFreeFuncForUserType(gt);
  si::insertStatementAfter(new_decl, free_decl);
  gt->aux_free_decl() = free_decl;

  // Build GridCopyin for this type
  SgFunctionDeclaration *copyin_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridCopyinFuncForUserType(gt);
  si::insertStatementAfter(free_decl, copyin_decl);
  gt->aux_copyin_decl() = copyin_decl;

  // Build GridCopyout for this type
  SgFunctionDeclaration *copyout_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridCopyoutFuncForUserType(gt);
  si::insertStatementAfter(copyin_decl, copyout_decl);
  gt->aux_copyout_decl() = copyout_decl;

  // Build GridGet for this type
  SgFunctionDeclaration *get_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridGetFuncForUserType(gt);
  si::insertStatementAfter(copyout_decl, get_decl);
  gt->aux_get_decl() = get_decl;

  SgFunctionDeclaration *emit_decl =
      static_cast<CUDARuntimeBuilder*>(rt_builder_)->
      BuildGridEmitFuncForUserType(gt);
  si::insertStatementAfter(get_decl, emit_decl);
  gt->aux_emit_decl() = emit_decl;
  return;
}


SgType *CUDATranslator::BuildOnDeviceGridType(GridType *gt) const {
  PSAssert(gt);
  // If this type is a user-defined type, its device type is stored at
  // gt->aux_type when building the type definition.
  if (gt->aux_type()) return gt->aux_type();
  
  string ondev_type_name = "__" + gt->type_name() + "_dev";
  LOG_DEBUG() << "On device grid type name: "
              << ondev_type_name << "\n";
  SgType *t =
      si::lookupNamedTypeInParentScopes(ondev_type_name, global_scope_);
  PSAssert(t);
  return t;
}

SgFunctionDeclaration *CUDATranslator::BuildRunKernel(StencilMap *stencil) {
  SgFunctionParameterList *params = sb::buildFunctionParameterList();
  SgClassDefinition *param_struct_def = stencil->GetStencilTypeDefinition();
  PSAssert(param_struct_def);

  SgInitializedName *dom_arg = NULL;
  // Build the parameter list for the function
  const SgDeclarationStatementPtrList &members =
      param_struct_def->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    SgName arg_name = vars[0]->get_name();
    SgType *arg_type = vars[0]->get_type();
    if (GridType::isGridType(arg_type)) {
      LOG_DEBUG() << "Grid type param\n";
      SgType *gt = BuildOnDeviceGridType(tx_->findGridType(arg_type));
      LOG_DEBUG() << "On dev type: " << gt->unparseToString() << "\n";
      arg_type = gt;
      // skip the grid index
      ++member;
    }
    LOG_DEBUG() << "type: " << arg_type->unparseToString() << "\n";    
    SgInitializedName *arg = sb::buildInitializedName(
        arg_name, arg_type);
    si::appendArg(params, arg);    
    if (Domain::isDomainType(arg_type) && dom_arg == NULL) {
      dom_arg = arg;
    }
  }
  PSAssert(dom_arg);

  // Append RB color param
  if (stencil->IsRedBlackVariant()) {
    SgInitializedName *rb_param =
        sb::buildInitializedName(PS_STENCIL_MAP_RB_PARAM_NAME,
                                 sb::buildIntType());
    si::appendArg(params, rb_param);
  }

  LOG_INFO() << "Declaring and defining function named "
             << stencil->getRunName() << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(stencil->getRunName(),
                                           sb::buildVoidType(),
                                           params, global_scope_);
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));
  // Make this function a CUDA global function
  LOG_DEBUG() << "Make the function a CUDA kernel\n";
  run_func->get_functionModifier().setCudaKernel();
  // Build and set the function body
  SgBasicBlock *func_body = BuildRunKernelBody(stencil, params);
  si::replaceStatement(run_func->get_definition()->get_body(),
                       func_body);
  // Mark this function as RunKernel
  rose_util::AddASTAttribute(run_func,
                             new RunKernelAttribute(stencil));
  si::fixVariableReferences(run_func);
  return run_func;
}

// Expresions themselves in index_args are used (no copy)
SgExprListExp *CUDATranslator::BuildKernelCallArgList(
    StencilMap *stencil,
    SgExpressionPtrList &index_args) {
  SgClassDefinition *stencil_def = stencil->GetStencilTypeDefinition();

  SgExprListExp *args = sb::buildExprListExp();
  FOREACH(it, index_args.begin(), index_args.end()) {
    si::appendExpression(args, *it);
  }
  
  // append the fields of the stencil type to the argument list  
  SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *var_decl = isSgVariableDeclaration(*it);
    PSAssert(var_decl);
    SgVariableDefinition *var_def = var_decl->get_definition();
    PSAssert(var_def);
    SgTypedefType *var_type = isSgTypedefType(var_def->get_type());    
    //SgExpression *exp = sb::buildVarRefExp(var_decl);
    SgExpression *exp = sb::buildVarRefExp(
        var_decl->get_variables()[0]->get_name());
    if (GridType::isGridType(var_type)) {
      exp = sb::buildAddressOfOp(exp);
      // skip the grid index field
      ++it;
    }
    si::appendExpression(args, exp);
  }

  return args;
}

SgFunctionCallExp *CUDATranslator::BuildKernelCall(
    StencilMap *stencil,
    SgExpressionPtrList &index_args) {
  SgExprListExp *args  = BuildKernelCallArgList(stencil, index_args);
  SgFunctionCallExp *func_call =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp(stencil->getKernel()), args);
  return func_call;
}

static SgVariableDeclaration *BuildIndexVarDeclaration(const char *name,
                                                       int dim,
                                                       SgExpression *init,
                                                       SgScopeStatement *block) {
  SgVariableDeclaration *index = sb::buildVariableDeclaration
      (name, sb::buildIntType(),
       init ? sb::buildAssignInitializer(init) : NULL,
       block);
  rose_util::AddASTAttribute<RunKernelIndexVarAttribute>(
      index, new RunKernelIndexVarAttribute(dim));
  si::appendStatement(index, block);
  return index;
}

SgBasicBlock* CUDATranslator::BuildRunKernelBody(
    StencilMap *stencil,
    SgFunctionParameterList *param) {
  LOG_DEBUG() << __FUNCTION__;
  SgInitializedName *dom_arg = param->get_args()[0];
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  
  SgExpressionPtrList index_args;
  SgVariableDeclaration *loop_index = NULL;
  int loop_dim = -1;
  int dim = stencil->getNumDim();  
  SgVariableDeclaration *indices[dim];
  
  // x = blockIdx.x * blockDim.x + threadIdx.x;
  SgExpression *init_x =
      sb::buildAddOp(sb::buildMultiplyOp(
          sbx::buildCudaIdxExp(sbx::kBlockIdxX),
          sbx::buildCudaIdxExp(sbx::kBlockDimX)),
                     sbx::buildCudaIdxExp(sbx::kThreadIdxX));
  if (stencil->IsRedBlackVariant()) {
    init_x = sb::buildMultiplyOp(init_x, sb::buildIntVal(2));
  }
  SgVariableDeclaration *x_index = BuildIndexVarDeclaration(
      LOOP_INDEX_VAR_NAME1, 1, init_x, block);
  index_args.push_back(sb::buildVarRefExp(x_index));
  indices[0] = x_index;
  

  if (dim >= 2) {
    SgVariableDeclaration *y_index = BuildIndexVarDeclaration(
        LOOP_INDEX_VAR_NAME2, 2,
        sb::buildAddOp(sb::buildMultiplyOp(
            sbx::buildCudaIdxExp(sbx::kBlockIdxY),
            sbx::buildCudaIdxExp(sbx::kBlockDimY)),
                       sbx::buildCudaIdxExp(sbx::kThreadIdxY)),
        block);
    index_args.push_back(sb::buildVarRefExp(y_index));
    indices[1] = y_index;
  }
  
  if (dim >= 3) {
    // y = blockIdx.y * blockDim.y + threadIdx.y;    
    SgVariableDeclaration *z_index = BuildIndexVarDeclaration(
        LOOP_INDEX_VAR_NAME3, 3, NULL, block);    
    index_args.push_back(sb::buildVarRefExp(z_index));
    loop_index = z_index;
    loop_dim = 2;
    indices[2] = z_index;
  }


  SgFunctionCallExp *kernel_call =
      BuildKernelCall(stencil, index_args);
  SgScopeStatement *kernel_call_block = block;

  if (dim == 1) {
    SgVariableDeclaration* t[] = {indices[0]};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 1);
    si::appendStatement(
        BuildDomainInclusionCheck(
            range_checking_idx, dom_arg, sb::buildReturnStmt()),
        block);
  } else if (dim == 2) {
    SgVariableDeclaration* t[] = {indices[0], indices[1]};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
    si::appendStatement(
        BuildDomainInclusionCheck(
            range_checking_idx, dom_arg, sb::buildReturnStmt()),
        block);
  } else if (dim == 3) {
    SgExpression *loop_begin =
        sb::buildPntrArrRefExp(
            BuildDomMinRef(sb::buildVarRefExp(dom_arg)),
            sb::buildIntVal(loop_dim));
    SgStatement *loop_init = sb::buildAssignStatement(
        sb::buildVarRefExp(loop_index), loop_begin);
    SgExpression *loop_end =
        sb::buildPntrArrRefExp(
            BuildDomMaxRef(sb::buildVarRefExp(dom_arg)),
            sb::buildIntVal(loop_dim));
    SgStatement *loop_test = sb::buildExprStatement(
        sb::buildLessThanOp(sb::buildVarRefExp(loop_index),
                            loop_end));
    index_args.push_back(sb::buildVarRefExp(loop_index));

    SgVariableDeclaration* t[] = {
      stencil->IsRedBlackVariant() ? NULL: indices[0], indices[1]};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
    si::appendStatement(
        BuildDomainInclusionCheck(
            range_checking_idx, dom_arg, sb::buildReturnStmt()),
        block);

    SgExpression *loop_incr =
        sb::buildPlusPlusOp(sb::buildVarRefExp(loop_index));
    kernel_call_block = sb::buildBasicBlock();
    SgStatement *loop
        = sb::buildForStatement(loop_init, loop_test,
                                loop_incr, kernel_call_block);
    si::appendStatement(loop, block);
    rose_util::AddASTAttribute(
        loop,
        new RunKernelLoopAttribute(3));
  } else {
    PSAssert(0);
  }
  
  si::appendStatement(sb::buildExprStatement(kernel_call),
                      kernel_call_block);

  if (stencil->IsRedBlackVariant()) {
    PSAssert(dim == 3); // Lower dimension not implemented yet.
    SgExpression *rb_offset_init =
        sb::buildAddOp(
            sb::buildVarRefExp(indices[0]),
            sb::buildBitAndOp(
                sb::buildAddOp(
                    sb::buildAddOp(sb::buildVarRefExp(indices[1]),
                                   sb::buildVarRefExp(indices[2])),
                    sb::buildVarRefExp(param->get_args().back())),
                sb::buildIntVal(1)));
    SgVariableDeclaration *x_index_rb =
        sb::buildVariableDeclaration(
            LOOP_INDEX_VAR_NAME1 "_rb", sb::buildIntType(),
            sb::buildAssignInitializer(rb_offset_init));
    
    SgVariableDeclaration* t[] = {x_index_rb};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 1);
    si::prependStatement(
        BuildDomainInclusionCheck(range_checking_idx, dom_arg,
                                  sb::buildContinueStmt()),
        kernel_call_block);

    si::prependStatement(x_index_rb, kernel_call_block);
    si::replaceExpression(index_args[0], sb::buildVarRefExp(x_index_rb));
  }
  
  return block;
}

SgIfStmt *CUDATranslator::BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgInitializedName *dom_arg, SgStatement *true_stmt)  const {
  // check x and y domain coordinates, like:
  // if (x < dom.local_min[0] || x >= dom.local_max[0] ||
  //     y < dom.local_min[1] || y >= dom.local_max[1]) {
  //   return;
  // }
  
  SgExpression *test_all = NULL;
  ENUMERATE (dim, index_it, indices.begin(), indices.end()) {
    // No check for the unit-stride dimension when red-black ordering
    // is used
    SgVariableDeclaration *idx = *index_it;
    // NULL indicates no check required.
    if (idx == NULL) continue;
    SgExpression *dom_min = sb::buildPntrArrRefExp(
        sb::buildDotExp(sb::buildVarRefExp(dom_arg),
                        sb::buildVarRefExp("local_min")),
        sb::buildIntVal(dim));
    SgExpression *dom_max = sb::buildPntrArrRefExp(
        sb::buildDotExp(sb::buildVarRefExp(dom_arg),
                        sb::buildVarRefExp("local_max")),
        sb::buildIntVal(dim));
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(sb::buildVarRefExp(idx), dom_min),
        sb::buildGreaterOrEqualOp(sb::buildVarRefExp(idx), dom_max));
    if (test_all) {
      test_all = sb::buildOrOp(test_all, test);
    } else {
      test_all = test;
    }
  }
  SgIfStmt *ifstmt =
      sb::buildIfStmt(test_all, true_stmt, NULL);
  return ifstmt;
}

void CUDATranslator::FixAST() {
  // Change the dummy grid type to the actual one even if AST
  // validation is disabled
  FixGridType();
  // Use the ROSE variable reference fix
  if (validate_ast_) {
    si::fixVariableReferences(project_);
  }
}

// Change dummy Grid type to real type so that the whole AST can be
// validated.
void CUDATranslator::FixGridType() {
  SgNodePtrList vdecls =
      NodeQuery::querySubTree(project_, V_SgVariableDeclaration);
  SgType *real_grid_type =
      si::lookupNamedTypeInParentScopes("__PSGrid");
  PSAssert(real_grid_type);
  FOREACH (it, vdecls.begin(), vdecls.end()) {
    SgVariableDeclaration *vdecl = isSgVariableDeclaration(*it);
    SgInitializedNamePtrList &vars = vdecl->get_variables();
    FOREACH (vars_it, vars.begin(), vars.end()) {
      SgInitializedName *var = *vars_it;
      if (GridType::isGridType(var->get_type())) {
        var->set_type(sb::buildPointerType(real_grid_type));
      }
    }
  }
  SgNodePtrList params =
      NodeQuery::querySubTree(project_, V_SgFunctionParameterList);
  FOREACH (it, params.begin(), params.end()) {
    SgFunctionParameterList *pl = isSgFunctionParameterList(*it);
    SgInitializedNamePtrList &vars = pl->get_args();
    FOREACH (vars_it, vars.begin(), vars.end()) {
      SgInitializedName *var = *vars_it;
      if (GridType::isGridType(var->get_type())) {
        var->set_type(sb::buildPointerType(real_grid_type));
      }
    }
  }
  
}

/** add dynamic parameter
 * @param[in/out] parlist ... parameter list
 */
void CUDATranslator::AddDynamicParameter(
    SgFunctionParameterList *parlist) {
  si::appendArg(parlist, sb::buildInitializedName("x", sb::buildIntType()));
  si::appendArg(parlist, sb::buildInitializedName("y", sb::buildIntType()));
  si::appendArg(parlist, sb::buildInitializedName("z", sb::buildIntType()));
}
/** add dynamic argument
 * @param[in/out] args ... arguments
 * @param[in] a_exp ... index expression
 */
void CUDATranslator::AddDynamicArgument(
    SgExprListExp *args, SgExpression *a_exp) {
  SgExpression *a =
      sb::buildPntrArrRefExp(
          sb::buildVarRefExp(
              sb::buildVariableDeclaration(
                  "__cuda_block_size",
                  sb::buildArrayType(cuda_block_size_type_))),
          a_exp);
  si::appendExpression(args, sb::buildDotExp(a, sb::buildVarRefExp("x")));
  si::appendExpression(args, sb::buildDotExp(a, sb::buildVarRefExp("y")));
  si::appendExpression(args, sb::buildDotExp(a, sb::buildVarRefExp("z")));
}
/** add some code after dlclose()
 * @param[in] scope
 */
void CUDATranslator::AddSyncAfterDlclose(
    SgScopeStatement *scope) {
  /* adHoc: cudaThreadSynchronize() need after dlclose().
   * if not, sometimes fail kernel calling.
   */
  si::appendStatement(
      sb::buildExprStatement(BuildCudaThreadSynchronize()),
      scope);
}

void CUDATranslator::TranslateFree(SgFunctionCallExp *node,
                                   GridType *gt) {
  LOG_DEBUG() << "Translating Free for CUDA\n";
  
  // Translate to __PSGridFree. Pass the specific free function for
  // user-defined point type
  SgExpression *free_func = NULL;
  if (gt->IsPrimitivePointType()) {
    free_func = rose_util::buildNULL(global_scope_);
  } else {
    free_func = sb::buildFunctionRefExp(gt->aux_free_decl());
  }

  SgExprListExp *args = isSgExprListExp(
      si::copyExpression(node->get_args()));
  si::appendExpression(args, free_func);
  SgFunctionCallExp *target = sb::buildFunctionCallExp(
      "__PSGridFree", sb::buildVoidType(),
      args);

  si::replaceExpression(node, target);
  return;
}

void CUDATranslator::TranslateCopyin(SgFunctionCallExp *node,
                                     GridType *gt) {
  LOG_DEBUG() << "Translating Copyin for CUDA\n";
  
  // Translate to __PSGridCopyin. Pass the specific copyin function for
  // user-defined point type
  SgExpression *func = NULL;
  if (gt->IsPrimitivePointType()) {
    func = rose_util::buildNULL(global_scope_);
  } else {
    func = sb::buildFunctionRefExp(gt->aux_copyin_decl());
  }

  SgExprListExp *args = isSgExprListExp(
      si::copyExpression(node->get_args()));
  si::appendExpression(args, func);
  SgFunctionCallExp *target = sb::buildFunctionCallExp(
      "__PSGridCopyin", sb::buildVoidType(),
      args);

  si::replaceExpression(node, target);
  return;
}

void CUDATranslator::TranslateCopyout(SgFunctionCallExp *node,
                                      GridType *gt) {
  LOG_DEBUG() << "Translating Copyout for CUDA\n";
  
  // Translate to __PSGridCopyout. Pass the specific copyin function for
  // user-defined point type
  SgExpression *func = NULL;
  if (gt->IsPrimitivePointType()) {
    func = rose_util::buildNULL(global_scope_);
  } else {
    func = sb::buildFunctionRefExp(gt->aux_copyout_decl());
  }

  SgExprListExp *args = isSgExprListExp(
      si::copyExpression(node->get_args()));
  si::appendExpression(args, func);
  SgFunctionCallExp *target = sb::buildFunctionCallExp(
      "__PSGridCopyout", sb::buildVoidType(),
      args);

  si::replaceExpression(node, target);
  return;
}


} // namespace translator
} // namespace physis

