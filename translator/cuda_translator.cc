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
  flag_pre_calc_grid_address_ = false;
  const pu::LuaValue *lv
      = config.Lookup(Configuration::CUDA_PRE_CALC_GRID_ADDRESS);
  if (lv) {
    PSAssert(lv->get(flag_pre_calc_grid_address_));
  }
  if (flag_pre_calc_grid_address_) {
    LOG_INFO() << "Optimization of address calculation enabled.\n";
  }
  // Redefine the block size if specified in the configuration file
  lv = config.Lookup(Configuration::CUDA_BLOCK_SIZE);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    std::vector<double> v;
    PSAssert(tbl->get(v));
    block_dim_x_ = (int)v[0];
    block_dim_y_ = (int)v[1];
    block_dim_z_ = (int)v[2];
  }
}

void CUDATranslator::run() {
  ReferenceTranslator::run();
}

void CUDATranslator::translateKernelDeclaration(
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

  
  Rose_STL_Container<SgNode*> gdim_calls =
      NodeQuery::querySubTree(node, V_SgFunctionCallExp);
  SgFunctionSymbol *gdim_dev =
      si::lookupFunctionSymbolInParentScopes("__PSGridDimDev");
  FOREACH (it, gdim_calls.begin(), gdim_calls.end()) {
    SgFunctionCallExp *fc = isSgFunctionCallExp(*it);
    PSAssert(fc);
    if (fc->getAssociatedFunctionSymbol() != grid_dim_get_func_)
      continue;
    fc->set_function(sb::buildFunctionRefExp(gdim_dev));
  }
  
  return;
}

void CUDATranslator::translateGet(SgFunctionCallExp *func_call_exp,
                                  SgInitializedName *grid_arg,
                                  bool is_kernel) {
  if (!(is_kernel && flag_pre_calc_grid_address_)) {
    ReferenceTranslator::translateGet(func_call_exp,
                                      grid_arg,
                                      is_kernel);
    return;
  }
  PSAssert(func_call_exp);
  PSAssert(grid_arg);

  SgFunctionDeclaration *func_decl = getContainingFunction(func_call_exp);
  PSAssert(func_decl);
  SgFunctionDefinition *func_def = func_decl->get_definition();
  PSAssert(func_def);
  SgBasicBlock *func_body = func_def->get_body();
  PSAssert(func_body);

  GridType *grid_type =
      tx_->findGridType(grid_arg->get_type());
  int num_dim = grid_type->getNumDim();

  SgVariableDeclaration *var_ptr =
      sb::buildVariableDeclaration(
          rose_util::generateUniqueName(func_body),
          sb::buildPointerType(grid_type->getElmType()),
          NULL,
          func_body);

  SgVarRefExp *grid_var = sb::buildVarRefExp(grid_arg, func_body);
  var_ptr->reset_initializer(
      sb::buildAssignInitializer(
          sb::buildAddOp(
              sb::buildCastExp(
                  sb::buildArrowExp(
                      grid_var,
                      sb::buildVarRefExp("p0",
                                         grid_decl_->get_definition())),
                  sb::buildPointerType(grid_type->getElmType())),
              buildOffset(grid_arg,
                          func_body,
                          num_dim,
                          func_call_exp->get_args()->get_expressions()))));
  func_body->prepend_statement(var_ptr);
  SgExpression *grid_get_exp =
      sb::buildPointerDerefExp(sb::buildVarRefExp(var_ptr));

  si::replaceExpression(func_call_exp, grid_get_exp);
}

SgVariableDeclaration *CUDATranslator::BuildGridDimDeclaration(
    const SgName &name,
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
  SgExpression *dim_y =
      sb::buildDivideOp(dom_dim_y,
                        sb::buildCastExp(block_dim_y,
                                         sb::buildDoubleType()));
  dim_y = BuildFunctionCall("ceil", dim_y);
  dim_y = sb::buildCastExp(dim_y, sb::buildIntType());
  SgExpression *dim_z = sb::buildIntVal(1);
  SgVariableDeclaration *grid_dim =
      sbx::buildDim3Declaration(name, dim_x, dim_y, dim_z, scope);
  return grid_dim;
}


SgExpression *CUDATranslator::BuildBlockDimX() {
  return sb::buildIntVal(block_dim_x_);
}

SgExpression *CUDATranslator::BuildBlockDimY() {
  return sb::buildIntVal(block_dim_y_);  
}

SgExpression *CUDATranslator::BuildBlockDimZ() {
  return sb::buildIntVal(block_dim_z_);
}

SgBasicBlock *CUDATranslator::BuildRunBody(Run *run) {
  SgBasicBlock *block = sb::buildBasicBlock();
  // int i;
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
  SgBasicBlock *loop_body = BuildRunLoopBody(run, block);
  SgForStatement *loop =
      sb::buildForStatement(loop_init, loop_test, loop_incr, loop_body);

  TraceStencilRun(run, loop, block);
  
  // cudaThreadSynchronize after each loop
  block->insert_statement(
      loop, sb::buildExprStatement(BuildCudaThreadSynchronize()), false);
  
  return block;
}

SgExprListExp *CUDATranslator::BuildCUDAKernelArgList(int stencil_idx,
                                                      StencilMap *sm,
                                                      SgVarRefExp *sv) const {
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
        sb::buildDotExp(sv, sb::buildVarRefExp(member_decl));
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
    args->append_expression(arg);
  }
  return args;
}

SgBasicBlock *CUDATranslator::BuildRunLoopBody(
    Run *run, SgScopeStatement *outer_block) {
  SgVariableDeclaration *block_dim =
      sbx::buildDim3Declaration("block_dim", BuildBlockDimX(),
                                BuildBlockDimY(),  BuildBlockDimZ(),
                                outer_block);
  outer_block->append_statement(block_dim);
  
  SgBasicBlock *loop_body = sb::buildBasicBlock();

  // Generates a call to each of the stencil function specified in the
  // PSStencilRun.
  ENUMERATE(stencil_idx, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *sm = it->second;    

    // Generate cache config code for each kernel
    SgFunctionSymbol *func_sym = rose_util::getFunctionSymbol(sm->run());
    PSAssert(func_sym);
    // Set the SM on-chip memory to prefer the L1 cache
    SgFunctionCallExp *cache_config =
        sbx::buildCudaCallFuncSetCacheConfig(func_sym,
                                             sbx::cudaFuncCachePreferL1);
    // Append invocation statement ahead of the loop
    outer_block->append_statement(sb::buildExprStatement(cache_config));

    string stencil_name = "s" + toString(stencil_idx);
    SgVarRefExp *sv = sb::buildVarRefExp(stencil_name);
    SgExprListExp *args = BuildCUDAKernelArgList(stencil_idx, sm, sv);    

    SgVariableDeclaration *grid_dim =
        BuildGridDimDeclaration(stencil_name + "_grid_dim",
                                BuildStencilDomMaxRef(sv, 0),
                                BuildStencilDomMaxRef(sv, 1),
                                BuildBlockDimX(),
                                BuildBlockDimY(),
                                outer_block);
    outer_block->append_statement(grid_dim);

    // Generate Kernel invocation code
    SgCudaKernelExecConfig *cuda_config =
        sbx::buildCudaKernelExecConfig(sb::buildVarRefExp(grid_dim),
                                       sb::buildVarRefExp(block_dim),
                                       NULL, NULL);
    SgCudaKernelCallExp *cuda_call =
        sbx::buildCudaKernelCallExp(sb::buildFunctionRefExp(func_sym),
                                    args, cuda_config);
    loop_body->append_statement(sb::buildExprStatement(cuda_call));
    appendGridSwap(sm, sv, loop_body);
  }
  return loop_body;
}

SgType *CUDATranslator::BuildOnDeviceGridType(GridType *gt) const {
  PSAssert(gt);
  string gt_name;
  int nd = gt->getNumDim();
  string elm_name = gt->getElmType()->unparseToString();
  std::transform(elm_name.begin(), elm_name.begin() +1,
                 elm_name.begin(), toupper);
  string ondev_type_name = "__PSGrid" + toString(nd) + "D"
                           + elm_name + "Dev";
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
    SgInitializedName *arg = new SgInitializedName(*vars[0]);
    SgType *type = arg->get_type();
    LOG_DEBUG() << "type: " << type->unparseToString() << "\n";
    if (Domain::isDomainType(type)) {
      if (!dom_arg) { dom_arg = arg; }
    } else if (GridType::isGridType(type)) {
      SgType *gt = BuildOnDeviceGridType(tx_->findGridType(type));
      arg->set_type(gt);
      // skip the grid index
      ++member;
    }
    params->append_arg(arg);
  }
  PSAssert(dom_arg);

  LOG_INFO() << "Declaring and defining function named "
             << stencil->getRunName() << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(stencil->getRunName(),
                                           sb::buildVoidType(),
                                           params, global_scope_);
  
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));
  // Make this function a CUDA global function
  SgFunctionModifier &modifier = run_func->get_functionModifier();
  modifier.setCudaKernel();
  // Build and set the function body
  SgBasicBlock *func_body = BuildRunKernelBody(stencil, dom_arg);
  run_func->get_definition()->set_body(func_body);

  return run_func;
}

SgExprListExp *CUDATranslator::BuildKernelCallArgList(
    StencilMap *stencil,
    SgExpressionPtrList &index_args) {
  SgClassDefinition *stencil_def = stencil->GetStencilTypeDefinition();

  SgExprListExp *args = sb::buildExprListExp();
  FOREACH(it, index_args.begin(), index_args.end()) {
    args->append_expression(*it);
  }
  
  // append the fields of the stencil type to the argument list  
  SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *var_decl = isSgVariableDeclaration(*it);
    PSAssert(var_decl);
    SgExpression *exp = sb::buildVarRefExp(var_decl);
    SgVariableDefinition *var_def = var_decl->get_definition();
    PSAssert(var_def);
    SgTypedefType *var_type = isSgTypedefType(var_def->get_type());
    if (GridType::isGridType(var_type)) {
      exp = sb::buildAddressOfOp(exp);
      // skip the grid index field
      ++it;
    }
    args->append_expression(exp);
  }

  return args;
}

SgFunctionCallExp *CUDATranslator::BuildKernelCall(
    StencilMap *stencil,
    SgExpressionPtrList &index_args) {
  SgExprListExp *args  = BuildKernelCallArgList(stencil, index_args);
  SgFunctionCallExp *func_call =
      sb::buildFunctionCallExp(
          rose_util::getFunctionSymbol(stencil->getKernel()), args);
  return func_call;
}

SgBasicBlock* CUDATranslator::BuildRunKernelBody(
    StencilMap *stencil,
    SgInitializedName *dom_arg) {
  LOG_DEBUG() << __FUNCTION__;
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  
  SgExpression *domain = sb::buildVarRefExp(dom_arg);
  SgExpression *min_field = BuildDomMinRef(domain);
  SgExpression *max_field = BuildDomMaxRef(domain);

  SgExpressionPtrList index_args;

  int dim = stencil->getNumDim();
  if (dim < 3) {
    LOG_ERROR() << "not supported yet.\n";
  } else if (dim == 3) {
    SgVariableDeclaration *x_index = sb::buildVariableDeclaration
        ("x", sb::buildIntType(), NULL, block);
    SgVariableDeclaration *y_index = sb::buildVariableDeclaration
        ("y", sb::buildIntType(), NULL, block);
    SgVariableDeclaration *z_index = sb::buildVariableDeclaration
        ("z", sb::buildIntType(), NULL, block);
    block->append_statement(y_index);
    block->append_statement(x_index);
    index_args.push_back(sb::buildVarRefExp(x_index));
    index_args.push_back(sb::buildVarRefExp(y_index));

    // x = blockIdx.x * blockDim.x + threadIdx.x;
    x_index->reset_initializer(
        sb::buildAssignInitializer(
            sb::buildAddOp(sb::buildMultiplyOp(
                sbx::buildCudaIdxExp(sbx::kBlockIdxX),
                sbx::buildCudaIdxExp(sbx::kBlockDimX)),
                           sbx::buildCudaIdxExp(sbx::kThreadIdxX))));
    // y = blockIdx.y * blockDim.y + threadIdx.y;
    y_index->reset_initializer(
        sb::buildAssignInitializer(
            sb::buildAddOp(sb::buildMultiplyOp(
                sbx::buildCudaIdxExp(sbx::kBlockIdxY),
                sbx::buildCudaIdxExp(sbx::kBlockDimY)),
                           sbx::buildCudaIdxExp(sbx::kThreadIdxY))));
    SgVariableDeclaration *loop_index = z_index;
    SgStatement *loop_init = sb::buildAssignStatement(
        sb::buildVarRefExp(loop_index),
        sb::buildPntrArrRefExp(min_field, sb::buildIntVal(2)));
    SgStatement *loop_test = sb::buildExprStatement(
        sb::buildLessThanOp(sb::buildVarRefExp(loop_index),
                            sb::buildPntrArrRefExp(max_field,
                                                   sb::buildIntVal(2))));
    index_args.push_back(sb::buildVarRefExp(loop_index));

    SgVariableDeclaration* t[] = {x_index, y_index};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
    block->append_statement(
        BuildDomainInclusionCheck(range_checking_idx, domain));
    
    block->append_statement(loop_index);

    SgExpression *loop_incr =
        sb::buildPlusPlusOp(sb::buildVarRefExp(loop_index));
    SgFunctionCallExp *kernel_call = BuildKernelCall(stencil, index_args);
    SgBasicBlock *loop_body =
        sb::buildBasicBlock(sb::buildExprStatement(kernel_call));
    SgStatement *loop
        = sb::buildForStatement(loop_init, loop_test, loop_incr, loop_body);
    block->append_statement(loop);
  }

  return block;
}

SgIfStmt *CUDATranslator::BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgExpression *dom_ref) const {
  // check x and y domain coordinates, like:
  // if (x < dom.local_min[0] || x >= dom.local_max[0] ||
  //     y < dom.local_min[1] || y >= dom.local_max[1]) {
  //   return;
  // }
  
  SgExpression *test_all = NULL;
  ENUMERATE (dim, index_it, indices.begin(), indices.end()) {
    SgExpression *idx = sb::buildVarRefExp(*index_it);
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(idx, BuildDomMinRef(dom_ref, dim)),
        sb::buildGreaterOrEqualOp(idx, BuildDomMaxRef(dom_ref, dim)));
    if (test_all) {
      test_all = sb::buildOrOp(test_all, test);
    } else {
      test_all = test;
    }
  }
  SgIfStmt *ifstmt =
      sb::buildIfStmt(test_all, sb::buildReturnStmt(), NULL);
  return ifstmt;
}

// void CUDATranslator::translateSet(SgFunctionCallExp *node,
//                                   SgInitializedName *gv) {
//   GridType *gt = tx_->findGridType(gv->get_type());
//   int nd = gt->getNumDim();
//   SgScopeStatement *scope = getContainingScopeStatement(node);    
//   SgExpression *offset = buildOffset(gv, scope, nd,
//                                      node->get_args()->get_expressions());
//   SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
//   SgExpression *p1 =
//       sb::buildArrowExp(g, sb::buildVarRefExp("p0",
//                                               grid_decl_->get_definition()));
//   p1 = sb::buildCastExp(p1, sb::buildPointerType(gt->getElmType()));
//   SgExpression *lhs = sb::buildPntrArrRefExp(p1, offset);
//   LOG_DEBUG() << "set lhs: " << lhs->unparseToString() << "\n";

//   SgExpression *rhs =
//       si::copyExpression(node->get_args()->get_expressions()[nd]);
  
//   LOG_DEBUG() << "set rhs: " << rhs->unparseToString() << "\n";

//   SgExpression *set = sb::buildAssignOp(lhs, rhs);
//   LOG_DEBUG() << "set: " << set->unparseToString() << "\n";

//   si::replaceExpression(node, set, false);
// }


} // namespace translator
} // namespace physis
