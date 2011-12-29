// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimization_passes.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

using std::vector;
using std::map;
using physis::translator::TranslationContext;
using physis::translator::RuntimeBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

/*
  Find groups of multiple indices that only differ at the loop
  dimension.  
 */
static vector<vector<StencilRegularIndexList> > FindCandidateIndices(
    const vector<StencilIndexList> &indices, int loop_dim) {
  map<StencilRegularIndexList, vector<StencilRegularIndexList> >
      grouped_indices;
  FOREACH (indices_it, indices.begin(), indices.end()) {
    StencilIndexList si = *indices_it;
    if (!StencilIndexRegularOrder(si)) continue;
    StencilRegularIndexList base_sim(si);
    base_sim.SetIndex(loop_dim, 0);
    if (!isContained(grouped_indices, base_sim)) {
      vector<StencilRegularIndexList> sv;
      grouped_indices.insert(make_pair(base_sim, sv));
    }
    vector<StencilRegularIndexList> &sim_vector =
        grouped_indices[base_sim];
    StencilRegularIndexList sim(si);
    if (!isContained(sim_vector, sim)) {
      sim_vector.push_back(sim);
    }
  }
  vector<vector<StencilRegularIndexList> > candidates;
  FOREACH (it, grouped_indices.begin(), grouped_indices.end()) {
    if (it->second.size() > 1) {
      vector<StencilRegularIndexList> x = it->second;
      std::sort(x.begin(), x.end());
      candidates.push_back(x);
#ifdef PS_DEBUG
      StringJoin sj;
      FOREACH (x_it, x.begin(), x.end()) {
        sj << *x_it;
      }
      LOG_DEBUG() << "Candidate: " << sj << "\n";
#endif
    }
  }
  LOG_DEBUG() << candidates.size() << " blocking candidate(s) found\n";
  return candidates;
}

static SgVariableDeclaration *BuildNewLocalVar(
    GridType *gt, TranslationContext *tx,
    SgScopeStatement *scope) {
  PSAssert(tx);
  PSAssert(scope);
  SgType *ty = gt->getElmType();
  string name = rose_util::generateUniqueName(scope);
  return sb::buildVariableDeclaration(name, ty, NULL, scope);
}

static SgExpression *ReplaceGetWithReg(SgForStatement *loop,
                                       SgInitializedName *gv,
                                       StencilRegularIndexList *index_list,
                                       SgVariableDeclaration *reg) {
  std::vector<SgNode*> gets =
      rose_util::QuerySubTreeAttribute<GridGetAttribute>(loop);
  SgExpression *replaced_get = NULL;
  FOREACH (it, gets.begin(), gets.end()) {
    SgExpression *get = isSgExpression(*it);
    PSAssert(get);
    GridGetAttribute *get_attr =
        rose_util::GetASTAttribute<GridGetAttribute>(get);
    if (get_attr->gv() != gv) continue;
    if (*index_list != get_attr->GetStencilIndexList())
      continue;
    if (!replaced_get) replaced_get = get;
    SgVarRefExp *reg_ref = sb::buildVarRefExp(reg);
    rose_util::CopyASTAttribute<GridGetAttribute>(reg_ref, get);    
    si::replaceExpression(get, reg_ref, true);
  }
  return replaced_get;
}

// Set loop var in run_kernel_attr.
static SgExpressionVector GetLoopIndices(
    SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop) {
  SgExpressionVector indices;
  RunKernelAttribute *run_kernel_attr =
      rose_util::GetASTAttribute<RunKernelAttribute>(run_kernel_func);
  indices.resize(run_kernel_attr->stencil_map()->getNumDim(), NULL);
  std::vector<SgNode*> index_vars = 
      rose_util::QuerySubTreeAttribute<RunKernelIndexVarAttribute>(
          run_kernel_func);
  FOREACH (index_vars_it, index_vars.begin(), index_vars.end()) {
    SgVariableDeclaration *index_var =
        isSgVariableDeclaration(*index_vars_it);
    RunKernelIndexVarAttribute *index_var_attr =
        rose_util::GetASTAttribute<RunKernelIndexVarAttribute>(index_var);
    indices.at(index_var_attr->dim()-1) = sb::buildVarRefExp(index_var);
  }
#ifdef PS_DEBUG
  StringJoin sj;
  FOREACH (indices_it, indices.begin(), indices.end()) {
    sj << ((*indices_it) ? (*indices_it)->unparseToString() : "<NULL>");
  }
  LOG_DEBUG() << "Indices: " << sj << "\n";
#endif
  return indices;
}

static void OffsetIndices(SgExpressionVector &indices,
                          StencilRegularIndexList &offset) {
  ENUMERATE (i, indices_it, indices.begin(), indices.end()) {
    ssize_t v = offset.GetIndex(i+1);
    if (v != 0) {
      *indices_it = sb::buildAddOp(*indices_it, sb::buildIntVal(v));
    }
  }
  return;
}

static void ReplaceOffsetIndices(SgFunctionCallExp *get_offset,
                                 SgExpressionVector &indices) {
  SgExprListExp *args = get_offset->get_args();
  SgExpressionPtrList &arg_exprs = args->get_expressions();
  LOG_DEBUG() << "Replacing " << get_offset->unparseToString() << "\n";
  ENUMERATE (i, arg_exprs_it, arg_exprs.begin() + 1, arg_exprs.end()) {
    LOG_DEBUG() << "Replacing "
                << (*arg_exprs_it)->unparseToString()
                << " with "
                << indices[i]->unparseToString() << "\n";
    si::replaceExpression(*arg_exprs_it, indices[i], true);
  }
}

static void ReplaceGetIndices(SgExpression *get,
                              SgExpressionVector &indices) {
  SgExpression *exp = rose_util::removeCasts(get);
  SgExpression *offset_exp = NULL;
  if (isSgPntrArrRefExp(exp)) {
    offset_exp = isSgPntrArrRefExp(exp)->get_rhs_operand();
  }
  PSAssert(offset_exp);

  if (isSgFunctionCallExp(offset_exp)) {
    ReplaceOffsetIndices(isSgFunctionCallExp(offset_exp), indices);
  } else {
    LOG_ERROR() << "Unsupported.\n";
    PSAbort(1);
  }
  
  return;
}

// Find the inner-most basic block of a given FOR body, which may have
// nested basic blocks.
// TODO: This does not work probably because of the incorrect AST
// manipulation. 
static SgBasicBlock *FindBasicBlock(SgStatement *for_body) {
  PSAssert(isSgBasicBlock(for_body));
  SgBasicBlock *bb = isSgBasicBlock(for_body);
  while (true) {
    // Assertion failure. This should not happen. Due to the incorrect
    // AST manipulation?
    PSAssert(si::getFirstStatement(bb));
    LOG_DEBUG() << "FIRST: " <<
        si::getFirstStatement(bb)->unparseToString() << "\n";
    if (isSgBasicBlock(si::getFirstStatement(bb))) {
      bb = isSgBasicBlock(si::getFirstStatement(bb));
    } else {
      break;
    }
  }
  LOG_DEBUG() << "bb: " << bb->unparseToString() << "\n";
  return bb;
}

static SgExpression *BuildGet(SgExpression *original_get,
                              SgInitializedName *gv,
                              SgFunctionDeclaration *run_kernel_func,
                              RuntimeBuilder *builder) {
  SgExpression *get = si::copyExpression(original_get);
  Rose_STL_Container<SgNode*> expressions =
      NodeQuery::querySubTree(get, V_SgExpression);
  SgExpression *gvref =
      builder->BuildGridRefInRunKernel(gv, run_kernel_func);
  FOREACH (expressions_it, expressions.begin(),
           expressions.end()) {
    SgExpression *exp = isSgExpression(*expressions_it);
    if (gvref->get_type() == exp->get_type()) {
      si::replaceExpression(exp, gvref);
      gvref = builder->BuildGridRefInRunKernel(gv, run_kernel_func);
    }
  }
  si::deleteAST(gvref);
  return get;
}

static void DoRegisterBlockingOneLine(
    TranslationContext *tx, RuntimeBuilder *builder,    
    SgInitializedName *gv,
    SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop,
    vector<StencilRegularIndexList> &bil) {
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  vector<SgVariableDeclaration*> registers;
  for (int i = 0; i < (int)bil.size(); ++i) {
    SgVariableDeclaration *reg = BuildNewLocalVar(
        rose_util::GetASTAttribute<GridType>(gv), tx,
        si::getScope(loop->get_parent()));
    registers.push_back(reg);
    si::insertStatementBefore(loop, reg);
  }
  vector<SgStatement*> move_stmts;  
  ENUMERATE (i, bil_it, bil.begin(), bil.end()) {
    StencilRegularIndexList il = *bil_it;
    SgVariableDeclaration *reg1 = registers[i];
    SgExpression *original_get =
        ReplaceGetWithReg(loop, gv, &il, reg1);
    if (i < (int)bil.size() - 1) {
      SgExpression *init_get =
          BuildGet(original_get, gv, run_kernel_func, builder);
      SgExpressionVector init_indices =
          GetLoopIndices(run_kernel_func, loop);
      si::deleteAST(init_indices[dim-1]);
      init_indices[dim-1] = si::copyExpression(loop_attr->begin());
      OffsetIndices(init_indices, il);
      ReplaceGetIndices(init_get, init_indices);
      registers[i+1]->reset_initializer(
          sb::buildAssignInitializer(init_get));
    }
    // Register move statements and a grid get statement for
    // the next point
    if (i < (int)bil.size() - 1) {
      SgStatement *reg_move =
          sb::buildAssignStatement(
              sb::buildVarRefExp(reg1),
              sb::buildVarRefExp(registers[i+1]));
      move_stmts.push_back(reg_move);
    } else {
      // Load a new value to reg
      SgExpression *get_next =
          BuildGet(original_get, gv, run_kernel_func, builder);
      SgExpressionVector indices_next =
          GetLoopIndices(run_kernel_func, loop);
      OffsetIndices(indices_next, il);
      ReplaceGetIndices(get_next, indices_next);
      SgExprStatement *asn_stmt =
          sb::buildAssignStatement(sb::buildVarRefExp(reg1),
                                   get_next);
      move_stmts.push_back(asn_stmt);
    }
  }
#if 0        
  SgBasicBlock *loop_body =
      FindBasicBlock(loop->get_loop_body());
#else
  SgScopeStatement *loop_body =
      isSgScopeStatement(loop->get_loop_body());
#endif        
  PSAssert(loop_body);
  FOREACH (move_stmts_it, move_stmts.rbegin(),
           move_stmts.rend()) {
    si::prependStatement(*move_stmts_it, loop_body);
    LOG_DEBUG() << "Prepending "
                << (*move_stmts_it)->unparseToString() << "\n";
  }
}
    
static void DoRegisterBlocking(
    SgProject *proj,
    TranslationContext *tx, RuntimeBuilder *builder,
    SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop) {
  RunKernelAttribute *run_kernel_attr =
      rose_util::GetASTAttribute<RunKernelAttribute>(run_kernel_func);
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  GridRangeMap &gr = run_kernel_attr->stencil_map()
      ->grid_stencil_range_map();
  FOREACH (gr_it, gr.begin(), gr.end()) {
    SgInitializedName *gv = gr_it->first;
    const vector<StencilIndexList> &indices = gr_it->second.all_indices();
    vector<vector<StencilRegularIndexList> > blocked_indices =
        FindCandidateIndices(indices, dim);
    FOREACH (blocked_indices_it, blocked_indices.begin(),
             blocked_indices.end()) {
      vector<StencilRegularIndexList> bil = *blocked_indices_it;
      DoRegisterBlockingOneLine(tx, builder, gv, run_kernel_func,
                                loop, bil);
    }
  }
}


void register_blocking(
    SgProject *proj,
    TranslationContext *tx,
    RuntimeBuilder *builder) {    
  pre_process(proj, tx, __FUNCTION__);

  std::vector<SgNode*> run_kernels =
      rose_util::QuerySubTreeAttribute<RunKernelAttribute>(proj);
  FOREACH (it, run_kernels.begin(), run_kernels.end()) {
    std::vector<SgNode*> run_kernel_loops =
        rose_util::QuerySubTreeAttribute<RunKernelLoopAttribute>(*it);
    size_t loop_number = run_kernel_loops.size();
    if (loop_number == 0) continue;
    SgForStatement *target_loop = NULL;
    FOREACH (run_kernel_loops_it, run_kernel_loops.rbegin(),
             run_kernel_loops.rend()) {
      SgForStatement *candidate_loop =
          isSgForStatement(*run_kernel_loops_it);
      RunKernelLoopAttribute *attr =
          rose_util::GetASTAttribute<RunKernelLoopAttribute>(
              candidate_loop);
      if (attr->IsMain()) {
        target_loop = candidate_loop;
        break;
      }
    }
    
    PSAssert(target_loop);
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(target_loop);
    LOG_DEBUG() << "Loop dimension: " << loop_attr->dim() << "\n";
    DoRegisterBlocking(proj, tx, builder, isSgFunctionDeclaration(*it),
                       target_loop);
  }
  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

