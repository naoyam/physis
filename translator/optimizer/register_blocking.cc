// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

#include <climits>

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

// Stencil may have "holes," but they also need to be blocked in
// registers.
// Eg., say a kernel has three gets: get(v, x-2), get(v, x), get(v,
// x+1). When applying register blocking, four registers are required:
// x-2, x-1, x, and x+1.

struct sri_comp {
  int dim;
  bool operator() (const StencilRegularIndexList &x,
                   const StencilRegularIndexList &y) {
    return x.GetIndex(dim) < y.GetIndex(dim);
  }
};

static void FillStencilHole(vector<vector<StencilRegularIndexList> >
                            &blocked_indices, int blocked_dim) {
  LOG_DEBUG() << "Blocked dim: " << blocked_dim << "\n";
  FOREACH (blocked_indices_it, blocked_indices.begin(),
           blocked_indices.end()) {
    vector<StencilRegularIndexList> &bil = *blocked_indices_it;
    int min_idx = INT_MAX, max_idx = INT_MIN;
    set<int> idx_set;
    FOREACH (bil_it, bil.begin(), bil.end()) {
      StencilRegularIndexList &sil = *bil_it;
      LOG_DEBUG() << "bil: " << sil << "\n";
      int index = sil.GetIndex(blocked_dim);
      idx_set.insert(index);
      min_idx = std::min(min_idx, index);
      max_idx = std::max(max_idx, index);
    }
    LOG_DEBUG() << "min: " << min_idx << "\n";
    LOG_DEBUG() << "max: " << max_idx << "\n";
    for (int idx = min_idx + 1; idx < max_idx; ++idx) {
      if (idx_set.find(idx) != idx_set.end()) continue;
      LOG_DEBUG() << "Missing index: " << idx << "\n";
      // copy StencilRegularIndexList
      StencilRegularIndexList missing_sil = *bil.begin();
      missing_sil.SetIndex(blocked_dim, idx);
      bil.push_back(missing_sil);
    }
    // Sort the StencilRegularIndexList vector to a increasing order 
    sri_comp comp;
    comp.dim = blocked_dim;
    std::sort(bil.begin(), bil.end(), comp);
    FOREACH (bil_it, bil.begin(), bil.end()) {
      LOG_DEBUG() << "bil filled: " << *bil_it << "\n";
    }
  }
  return;
}




static SgVariableDeclaration *BuildNewLocalVar(
    GridType *gt, TranslationContext *tx,
    SgScopeStatement *scope) {
  PSAssert(tx);
  PSAssert(scope);
  SgType *ty = gt->point_type();
  string name = rose_util::generateUniqueName(scope);
  return sb::buildVariableDeclaration(name, ty, NULL, scope);
}

static bool IsSameGrid(SgName name, SgInitializedName *grid) {
  // grid is a variable renamed when inlined. Assignment form varies
  // depending on the target.
  SgExpression *asini = isSgAssignInitializer(
      grid->get_initptr())->get_operand();
  SgVarRefExp *gv = NULL;
  if (isSgBinaryOp(asini)) {
    gv = isSgVarRefExp(
        isSgBinaryOp(asini)->get_rhs_operand());
    PSAssert(gv);
  } else if (isSgAddressOfOp(asini)) {
    gv = isSgVarRefExp(isSgAddressOfOp(asini)->get_operand());
    PSAssert(gv);    
  } else {
    LOG_ERROR() << "Unsupported grid representation: "
                << asini->unparseToString() << "\n";
    PSAbort(1);
  }
    
  return name == gv->get_symbol()->get_name();
}

static SgExpression *FindGetAtIndex(SgForStatement *loop,
                                    StencilRegularIndexList *index_list) {
  std::vector<SgNode*> gets =
      rose_util::QuerySubTreeAttribute<GridGetAttribute>(loop);
  FOREACH (it, gets.begin(), gets.end()) {
    SgExpression *get = isSgExpression(*it);
    PSAssert(get);
    GridGetAttribute *get_attr =
        rose_util::GetASTAttribute<GridGetAttribute>(get);
    if (get_attr->GetStencilIndexList() &&
        *index_list == *get_attr->GetStencilIndexList())
      return get;
  }
  return NULL;
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
    if (get_attr->GetStencilIndexList() == NULL) continue;
    if (*index_list != *get_attr->GetStencilIndexList())
      continue;
    // TODO: gv needs to be replaced with GridObject
    if (!IsSameGrid(gv->get_name(), get_attr->gv())) continue;
    if (!replaced_get) replaced_get = get;
    SgVarRefExp *reg_ref = sb::buildVarRefExp(reg);
    rose_util::CopyASTAttribute<GridGetAttribute>(reg_ref, get);    
    si::replaceExpression(get, reg_ref, true);
  }
  return replaced_get;
}

// Set loop var in run_kernel_attr.
static SgExpressionPtrList GetLoopIndices(
    SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop) {
  SgExpressionPtrList indices;
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

static SgExpression *ReplaceGridInGet(SgExpression *original_get,
                                      SgExpression *gvref) {
  SgExpression *get = si::copyExpression(original_get);
  Rose_STL_Container<SgNode*> expressions =
      NodeQuery::querySubTree(get, V_SgExpression);
  FOREACH (expressions_it, expressions.begin(),
           expressions.end()) {
    SgExpression *exp = isSgExpression(*expressions_it);
    if (gvref->get_type() == exp->get_type()) {
      si::replaceExpression(exp, si::copyExpression(gvref));
    }
  }
  return get;
}

#if 0
static SgExpression *GetOffsetFromGridGet(SgExpression *grid_get) {
  if (isSgPntrArrRefExp(grid_get)) {
    return isSgPntrArrRefExp(grid_get)->get_rhs_operand();
  }
  
  LOG_ERROR() << "Unsupported: " << grid_get->unparseToString() << "\n";
  PSAbort(1);
  return NULL;
}
#endif

static bool ShouldGetPeriodic(SgForStatement *loop,
                              vector<StencilRegularIndexList> &bil,
                              int dim, int index) {
  if (index == 0) return false;
  FOREACH (bil_it, bil.begin(), bil.end()) {
    StencilRegularIndexList &il = *bil_it;
    if (!((index > 0 && il.GetIndex(dim) > 0) ||
          (index < 0 && il.GetIndex(dim) < 0))) {
      continue;
    }
    SgExpression *get = FindGetAtIndex(loop, &il);
    // This index is originally a hole if no get is found.
    if (get == NULL) continue;
    bool p = rose_util::GetASTAttribute<GridOffsetAttribute>(
        rose_util::GetASTAttribute<GridGetAttribute>(
            get)->offset())->periodic();
    if (p) return true;
  }
  return false;
}

static void SetStencilIndexList(StencilIndexList &sil,
                                const StencilRegularIndexList &sril) {
  for (int i = 1; i <= sril.GetNumDims(); ++i) {
    sil.push_back(StencilIndex(i, sril.GetIndex(i)));
  }
}

static void DoRegisterBlockingOneLine(
    TranslationContext *tx, RuntimeBuilder *builder,    
    SgInitializedName *gv, SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop, vector<StencilRegularIndexList> &bil) {
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  GridType *gt = rose_util::GetASTAttribute<GridType>(gv);
  int dim = loop_attr->dim();
  vector<SgVariableDeclaration*> registers;
  // Declare local variables for register blocking
  for (int i = 0; i < (int)bil.size(); ++i) {
    SgVariableDeclaration *reg = BuildNewLocalVar(
        gt, tx, si::getEnclosingFunctionDefinition(loop));
    registers.push_back(reg);
    si::insertStatementBefore(loop, reg);
  }
  vector<SgStatement*> move_stmts;
  //bool backward_periodic = false, forward_periodic = false;
  ENUMERATE (i, bil_it, bil.begin(), bil.end()) {
    StencilRegularIndexList il = *bil_it;
    SgVariableDeclaration *reg1 = registers[i];
    SgExpression *original_get =
        ReplaceGetWithReg(loop, gv, &il, reg1);
    // There can be no get for this index. The element at that index
    // is still needed to be loaded for blocking.
    //PSAssert(original_get);
    // Whether this index should be accessed with the periodic
    // boundary condition can't be determined if this index is
    // originally a stencil hole. Conservatively determine whether it
    // should be periodic or not.
    bool is_periodic;
    if (original_get) {
#if 0
      is_periodic = rose_util::GetASTAttribute<GridOffsetAttribute>(
          GetOffsetFromGridGet(original_get))->periodic();
#else
      is_periodic = rose_util::GetASTAttribute<GridOffsetAttribute>(
        rose_util::GetASTAttribute<GridGetAttribute>(
            original_get)->offset())->periodic();
#endif      
    } else {
      is_periodic = ShouldGetPeriodic(loop, bil, dim, il.GetIndex(dim));
    }
    StencilIndexList sil;
    SetStencilIndexList(sil, il);
    // Initial load from memory to registers
    if (i < (int)bil.size() - 1) {
      SgExpressionPtrList init_indices =
          GetLoopIndices(run_kernel_func, loop);
      si::deleteAST(init_indices[dim-1]);
      init_indices[dim-1] = si::copyExpression(loop_attr->begin());
      OffsetIndices(init_indices, il);
      SgExpression *init_get = builder->BuildGridGet(
          builder->BuildGridRefInRunKernel(gv, run_kernel_func),
          gt, &init_indices, &sil,
          true, is_periodic);
      registers[i+1]->reset_initializer(
          sb::buildAssignInitializer(init_get));
    }
    // Insert move statements and a grid get statement for
    // the next point
    if (i < (int)bil.size() - 1) {
      SgStatement *reg_move =
          sb::buildAssignStatement(
              sb::buildVarRefExp(reg1),
              sb::buildVarRefExp(registers[i+1]));
      move_stmts.push_back(reg_move);
    } else {
      // Load a new value to reg
      SgExpressionVector indices_next =
          GetLoopIndices(run_kernel_func, loop);
      OffsetIndices(indices_next, il);
      SgExpression *get_next = builder->BuildGridGet(
          builder->BuildGridRefInRunKernel(gv, run_kernel_func),
          gt, &indices_next, &sil,
          true, is_periodic);
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
    FillStencilHole(blocked_indices, dim);
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

  vector<SgForStatement*> target_loops = FindInnermostLoops(proj);

  FOREACH (it, target_loops.begin(), target_loops.end()) {
    SgForStatement *target_loop = *it;
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(target_loop);
    if (!loop_attr->IsMain()) continue;
    LOG_DEBUG() << "Loop dimension: " << loop_attr->dim() << "\n";
    SgFunctionDeclaration *run_kernel_func =
        si::getEnclosingFunctionDeclaration(target_loop);
    DoRegisterBlocking(proj, tx, builder, run_kernel_func,
                       target_loop);
  }

  post_process(proj, tx, __FUNCTION__);
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

