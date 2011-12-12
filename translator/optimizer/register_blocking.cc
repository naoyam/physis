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

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

static vector<vector<StencilRegularIndexList> > FindCandidateIndices(
    const vector<StencilIndexList> &indices, int loop_dim) {
  map<StencilRegularIndexList, vector<StencilRegularIndexList> > grouped_indices;
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
  LOG_DEBUG() << "ReplaceGetWithReg\n";
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
    si::replaceExpression(get, reg_ref, true);
    rose_util::CopyASTAttribute<GridGetAttribute>(reg_ref, get);
  }
  return replaced_get;
}

static void DoRegisterBlocking(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop) {
  RunKernelAttribute *run_kernel_attr =
      rose_util::GetASTAttribute<RunKernelAttribute>(run_kernel_func);
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  StencilMap *sm = run_kernel_attr->stencil_map();
  GridRangeMap &gr = sm->grid_stencil_range_map();
  FOREACH (gr_it, gr.begin(), gr.end()) {
    SgInitializedName *gv = gr_it->first;
    GridType *gt = rose_util::GetASTAttribute<GridType>(gv);
    const StencilRange &sr = gr_it->second;
    const vector<StencilIndexList> &indices = sr.all_indices();
    vector<vector<StencilRegularIndexList> > blocked_indices =
        FindCandidateIndices(indices, dim);
    FOREACH (blocked_indices_it, blocked_indices.begin(),
             blocked_indices.end()) {
      vector<StencilRegularIndexList> bil = *blocked_indices_it;
      vector<SgVariableDeclaration*> registers;
      FOREACH (bil_it, bil.begin(), bil.end()) {
        SgVariableDeclaration *reg = BuildNewLocalVar(
            gt, tx, si::getScope(loop->get_parent()));
        StencilRegularIndexList il = *bil_it;
        si::insertStatementBefore(loop, reg);
        ReplaceGetWithReg(loop, gv, &il, reg);
      }
    }
  }
}


void register_blocking(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {    
  pre_process(proj, tx, __FUNCTION__);

  std::vector<SgNode*> run_kernels =
      rose_util::QuerySubTreeAttribute<RunKernelAttribute>(proj);
  FOREACH (it, run_kernels.begin(), run_kernels.end()) {
    std::vector<SgNode*> run_kernel_loops =
        rose_util::QuerySubTreeAttribute<RunKernelLoopAttribute>(*it);
    if (run_kernel_loops.size() == 0) continue;
    SgForStatement *loop =
        isSgForStatement(run_kernel_loops.back());
    PSAssert(loop);
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
    LOG_DEBUG() << "Loop dimension: " << loop_attr->dim() << "\n";
    DoRegisterBlocking(proj, tx, isSgFunctionDeclaration(*it),
                       loop);
  }
  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

