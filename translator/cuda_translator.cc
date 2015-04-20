// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/cuda_translator.h"

#include <algorithm>
#include <string>

#include "translator/cuda_runtime_builder.h"
#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/cuda_util.h"
#include "translator/stencil_analysis.h"
#include "translator/physis_names.h"

namespace pu = physis::util;
namespace sb = SageBuilder;
namespace si = SageInterface;
namespace ru = physis::translator::rose_util;
namespace cu = physis::translator::cuda_util;

namespace physis {
namespace translator {

CUDATranslator::CUDATranslator(const Configuration &config):
    ReferenceTranslator(config) {
  target_specific_macro_ = "PHYSIS_CUDA";
}

void CUDATranslator::SetUp(SgProject *project,
                           TranslationContext *context,
                           BuilderInterface *rt_builder) {
  ReferenceTranslator::SetUp(project, context, rt_builder);

  /* auto tuning & has dynamic arguments */
  if (!(config_.auto_tuning() && config_.ndynamic() > 1)) return;
  /* build __cuda_block_size_struct */
  SgClassDeclaration *s =
      sb::buildStructDeclaration(
          SgName("__cuda_block_size_struct"), global_scope_);
  SgClassDefinition *s_def = s->get_definition();
  si::appendStatement(
      sb::buildVariableDeclaration("x", sb::buildIntType(), NULL, s_def),
      s_def);
  si::appendStatement(
      sb::buildVariableDeclaration("y", sb::buildIntType(), NULL, s_def),
      s_def);
  si::appendStatement(
      sb::buildVariableDeclaration("z", sb::buildIntType(), NULL, s_def),
      s_def);
  cuda_block_size_type_ = s->get_type();
  dynamic_cast<CUDARuntimeBuilder*>(builder())->
      cuda_block_size_type() = cuda_block_size_type_;
  SgVariableDeclaration *cuda_block_size =
      sb::buildVariableDeclaration(
          "__cuda_block_size",
          sb::buildConstType(
              sb::buildArrayType(cuda_block_size_type_)),
          sb::buildAggregateInitializer(
              sb::buildExprListExp(dynamic_cast<CUDARuntimeBuilder*>(builder())->
                                   cuda_block_size_vals())),
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
    extra_arg = ru::BuildNULL();
  } else {
    extra_arg = sb::buildFunctionRefExp(gt->aux_new_decl());
  }
  si::appendExpression(args, extra_arg);
  return;
}

void CUDATranslator::TranslateKernelDeclaration(
    SgFunctionDeclaration *node) {
  cu::SetCUDADevice(node);

  // e.g., PSGrid3DFloat -> __PSGrid3DFloatDev *
  Rose_STL_Container<SgNode*> exps =
      NodeQuery::querySubTree(node, V_SgInitializedName);
  FOREACH (it, exps.begin(), exps.end()) {
    SgInitializedName *exp = isSgInitializedName(*it);
    PSAssert(exp);
    SgType *cur_type = exp->get_type();
    GridType *gt = ru::GetASTAttribute<GridType>(cur_type);
    // not a grid type
    if (!gt) continue;
    SgType *new_type = sb::buildPointerType(builder()->BuildOnDeviceGridType(gt));
    exp->set_type(new_type);
  }
  return;
}

void CUDATranslator::TranslateGetForUserDefinedType(
    SgDotExp *node, SgPntrArrRefExp *array_top) {
  SgVarRefExp *member_ref =
      isSgVarRefExp(node->get_rhs_operand());
  PSAssert(member_ref);
  SgExpression *get_exp = node->get_lhs_operand();
  GridGetAttribute *gga =
      rose_util::GetASTAttribute<GridGetAttribute>(
          get_exp);
  const string &mem_name = ru::GetName(member_ref);
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
        builder()->BuildGridGet(
            sb::buildVarRefExp(gv->get_name(),
                               si::getScope(node)),
            ru::GetASTAttribute<GridVarAttribute>(gv),
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
        builder()->BuildGridGet(
            sb::buildVarRefExp(gv->get_name(),
                               si::getScope(node)),
            ru::GetASTAttribute<GridVarAttribute>(gv),            
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
  GridType *gt = ru::GetASTAttribute<GridType>(grid_arg->get_type());
  SgExpressionPtrList args;
  rose_util::CopyExpressionPtrList(
      func_call_exp->get_args()->get_expressions(), args);
  const StencilIndexList *sil =
      rose_util::GetASTAttribute<GridGetAttribute>(
          func_call_exp)->GetStencilIndexList();
  SgExpression *gv = sb::buildVarRefExp(grid_arg->get_name(),
                                        si::getScope(func_call_exp));
  SgExpression *real_get =
      builder()->BuildGridGet(
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
  GridType *gt = ru::GetASTAttribute<GridType>(gv->get_type());
  
  // Just call the reference implementation if this grid is of a
  // primitive type. 
  if (gt->IsPrimitivePointType()) {
    ReferenceTranslator::TranslateEmit(node, attr);
    return;
  }

  // The type of the grid is a user-defined type.
  
  // build a function call to gt->aux_emit_decl(g, offset, v)
  int nd = gt->rank();
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
      builder()->BuildGridEmit(
          sb::buildVarRefExp(attr->gv()), attr, &args, v, si::getScope(node));
  
  si::replaceExpression(node, real_exp);

  if (!is_grid_type_specific_call) {
    RemoveEmitDummyExp(real_exp);
  }
  
}


void CUDATranslator::ProcessUserDefinedPointType(
    SgClassDeclaration *grid_decl, GridType *gt) {
  LOG_DEBUG() << "Define grid data type for device.\n";
  SgClassDeclaration *type_decl =
      builder()->BuildGridDevTypeForUserType(grid_decl, gt);
  LOG_DEBUG() << "Inserting device type for user type\n";
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
      builder()->BuildGridNewFuncForUserType(gt);
  LOG_DEBUG() << "Inserting new function for user type\n";
  si::insertStatementAfter(type_decl, new_decl);
  gt->aux_new_decl() = new_decl;
  // Build GridFree for this type
  SgFunctionDeclaration *free_decl =
      builder()->BuildGridFreeFuncForUserType(gt);
  LOG_DEBUG() << "Inserting free function for user type\n";
  si::insertStatementAfter(new_decl, free_decl);
  gt->aux_free_decl() = free_decl;

  // Build GridCopyin for this type
  SgFunctionDeclaration *copyin_decl =
      builder()->BuildGridCopyFuncForUserType(gt, false);
  LOG_DEBUG() << "Inserting copyin function for user type\n";
  si::insertStatementAfter(free_decl, copyin_decl);
  gt->aux_copyin_decl() = copyin_decl;

  // Build GridCopyout for this type
  SgFunctionDeclaration *copyout_decl =
      builder()->BuildGridCopyFuncForUserType(gt, true);
  LOG_DEBUG() << "Inserting copyout function for user type\n";  
  si::insertStatementAfter(copyin_decl, copyout_decl);
  gt->aux_copyout_decl() = copyout_decl;

  // Build GridGet for this type
  SgFunctionDeclaration *get_decl =
      builder()->BuildGridGetFuncForUserType(gt);
  LOG_DEBUG() << "Inserting get function for user type\n";  
  si::insertStatementAfter(copyout_decl, get_decl);
  gt->aux_get_decl() = get_decl;

  SgFunctionDeclaration *emit_decl =
      builder()->BuildGridEmitFuncForUserType(gt);
  LOG_DEBUG() << "Inserting emit function for user type\n";  
  si::insertStatementAfter(get_decl, emit_decl);
  gt->aux_emit_decl() = emit_decl;
  return;
}

void CUDATranslator::FixAST() {
  // Change the dummy grid type to the actual one even if AST
  FixGridType(string(grid_type_name_));
  if (getenv("PHYSISC_NO_FIX_VARIABLE_REFERENCES")) {
    LOG_INFO() << "Skipping variable reference fixing\n";
  } else {
    si::fixVariableReferences(project_);
  }
}

// Change dummy Grid type to real type so that the whole AST can be
// validated.
void CUDATranslator::FixGridType(const string &real_type_name) {
  SgNodePtrList vdecls =
      NodeQuery::querySubTree(project_, V_SgVariableDeclaration);
  SgType *real_grid_type =
      si::lookupNamedTypeInParentScopes(real_type_name);
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


void CUDATranslator::TranslateFree(SgFunctionCallExp *node,
                                   GridType *gt) {
  LOG_DEBUG() << "Translating Free for CUDA\n";
  
  // Translate to __PSGridFree. Pass the specific free function for
  // user-defined point type
  SgExpression *free_func = NULL;
  if (gt->IsPrimitivePointType()) {
    free_func = ru::BuildNULL();
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
    func = ru::BuildNULL();
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
    func = ru::BuildNULL();
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

