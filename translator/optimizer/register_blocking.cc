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

typedef vector<StencilRegularIndexList> IndexListVector;

//! Struct to represent a grid variable and a member to access.
struct GridData {
  SgInitializedName *gv_;
  bool member_access_;
  string member_;
  GridData(SgInitializedName *gv,
           string member):
      gv_(gv), member_access_(true),
      member_(member) {}
  explicit GridData(SgInitializedName *gv):
      gv_(gv), member_access_(false) {}
  GridData(const GridData &gd):
      gv_(gd.gv_),
      member_access_(gd.member_access_),
      member_(gd.member_) {}
};

/*!
  Find groups of multiple indices that only differ at the loop
  dimension.  
 */
static vector<IndexListVector> FindCandidateIndices(
    const vector<StencilIndexList> &indices, int loop_dim) {
  map<StencilRegularIndexList, IndexListVector>
      grouped_indices;
  FOREACH (indices_it, indices.begin(), indices.end()) {
    StencilIndexList si = *indices_it;
    if (!StencilIndexRegularOrder(si)) continue;
    StencilRegularIndexList base_sim(si);
    base_sim.SetIndex(loop_dim, 0);
    if (!isContained(grouped_indices, base_sim)) {
      IndexListVector sv;
      grouped_indices.insert(make_pair(base_sim, sv));
    }
    IndexListVector &sim_vector =
        grouped_indices[base_sim];
    StencilRegularIndexList sim(si);
    if (!isContained(sim_vector, sim)) {
      sim_vector.push_back(sim);
    }
  }
  vector<IndexListVector> candidates;
  FOREACH (it, grouped_indices.begin(), grouped_indices.end()) {
    if (it->second.size() > 1) {
      IndexListVector x = it->second;
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

static void FillStencilHole(
    vector<IndexListVector> &blocked_indices,
    int blocked_dim) {
  LOG_DEBUG() << "Blocked dim: " << blocked_dim << "\n";
  FOREACH (blocked_indices_it, blocked_indices.begin(),
           blocked_indices.end()) {
    IndexListVector &bil = *blocked_indices_it;
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
    SgType *ty, TranslationContext *tx,
    SgScopeStatement *scope) {
  PSAssert(tx);
  PSAssert(scope);
  string name = rose_util::generateUniqueName(scope);
  //return sb::buildVariableDeclaration(name, ty, NULL, scope);
  return sb::buildVariableDeclaration(name, ty);
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

//!
/*!
  \param loop
  \param gd
  \param index_list
  \param reg 
  \return one of replaced get expression.
 */
static SgExpression *ReplaceGetWithReg(SgForStatement *loop,
                                       const GridData &gd,
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
    //if (!IsSameGrid(gd.gv_->get_name(), get_attr->gv())) continue;
    if (gd.gv_ != get_attr->original_gv()) {
      LOG_DEBUG() << "NOT Same grid: "
                  << gd.gv_->get_name()
                  << " != "
                  << get_attr->original_gv() << "\n";
      continue;
    }
    // Ensure the get access is to the same member if its member
    // access 
    if (gd.member_access_) {
      if (get_attr->member_name() != gd.member_) continue;
    } else {
      // Ignore a get with a member access.
      // NOTE: Not handled because this shouldn't occur.
      if (get_attr->member_name() != "") continue;
    }
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

static bool ShouldGetPeriodic(SgForStatement *loop,
                              IndexListVector &bil,
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

//! Get type of a grid data.
/*!
  If it's an access to member, returns its member type; otherwise, the
  grid point type, which may be a struct type.
  
  \param gt
  \param gd
 */
static SgType *GetType(GridType *gt, const GridData &gd) {
  if (gt->IsPrimitivePointType() || !gd.member_access_) {
    return gt->point_type();
  } else {
    SgClassDefinition *utdef = gt->point_def();
    vector<SgVariableDeclaration*> mdecls =
        si::querySubTree<SgVariableDeclaration>(
            utdef, V_SgVariableDeclaration);
    FOREACH (it, mdecls.begin(), mdecls.end()) {
      SgVariableDeclaration *mdecl = *it;
      if (rose_util::GetName(mdecl) == gd.member_) {
        return mdecl->get_definition()->get_type();
      }
    }
    LOG_ERROR() << "No such member found: " << gd.member_
                << "\n";
    return NULL;
  }
}

SgExpression *BuildGet(const GridData &gd,
                       SgFunctionDeclaration *run_kernel_func,
                       GridType *gt,
                       const SgExpressionPtrList &indices,
                       const StencilIndexList &sil,
                       bool is_periodic,
                       RuntimeBuilder *builder) {
  SgInitializedName *gv = gd.gv_;
  SgExpression *grid_ref =
      builder->BuildGridRefInRunKernel(
          gv, run_kernel_func);
  return gd.member_access_? 
      builder->BuildGridGet(grid_ref, gt, &indices,
                            &sil, true, is_periodic,
                            gd.member_) :
      builder->BuildGridGet(grid_ref, gt, &indices,
                            &sil, true, is_periodic);
}

//! Apply register blocking along one line of accesses.
/*!
  For example, suppose a stencil:
  
  PSGridGet(x,0,0,-1)+PSGridGet(x,0,0,0)+PSGridGet(x,0,0,1)

  In this case, if the innner most loop is for the last dimension,
  they are replaced with local variable accesses. Parameter bil will
  contain a list, {(0, 0, -1), (0, 0, 0), (0, 0, +1)}, in this case.
  
  \param tx
  \param builder
  \param gd Accessed grid
  \param run_kernel_func Target function
  \param loop Target loop
  \param bil List of target access indices
 */
static bool DoRegisterBlockingOneLine(
    TranslationContext *tx, RuntimeBuilder *builder,    
    const GridData &gd, SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop, IndexListVector &bil) {
  /*
    Steps:
    1. Declare local variables to hold loaded grid elements
    2. Find and replace target grid accessses
    3. If replaced, insert the local variable declarations and their
    corresponding copy statements.

    Note that replacement may not necessarily happen. For example,
    suppose G be a grid with a user-defined point type, and the
    translator generates code with the AoS data layout. If gd
    specifies a member in the user-defined point type, no replacement
    is performed since all accesses grid data struct by struct rather
    than element by element.
   */
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  SgInitializedName *gv = gd.gv_;
  GridType *gt = rose_util::GetASTAttribute<GridType>(gv);
  int dim = loop_attr->dim();
  // Flag to indicate access replacement 
  bool replaced = false;
  // Local variables
  vector<SgVariableDeclaration*> registers;
  // Declare local variables for register blocking
  for (int i = 0; i < (int)bil.size(); ++i) {
    SgType *ty = GetType(gt, gd);
    // Do not insert the declarations yet since they may not be used.
    SgVariableDeclaration *reg = BuildNewLocalVar(
        ty, tx, si::getEnclosingFunctionDefinition(loop));
    registers.push_back(reg);
  }
  vector<SgStatement*> move_stmts;
  // Loop for each of the target stencil offset
  ENUMERATE (i, bil_it, bil.begin(), bil.end()) {
    // Replace all accesses with il offset by reg1
    StencilRegularIndexList il = *bil_it;
    SgVariableDeclaration *reg1 = registers[i];
    // Attempt to replace
    SgExpression *original_get =
        ReplaceGetWithReg(loop, gd, &il, reg1);
    // Track if any replacement is done
    replaced |= original_get != NULL;
    // There can be no get for this index. The element at that index
    // is still needed to be loaded for blocking.
    
    // Whether this index should be accessed with the periodic
    // boundary condition can't be determined if this index is
    // originally a stencil hole. Conservatively determine whether it
    // should be periodic or not.
    bool is_periodic;
    if (original_get) {
      is_periodic = rose_util::GetASTAttribute<GridOffsetAttribute>(
        rose_util::GetASTAttribute<GridGetAttribute>(
            original_get)->offset())->periodic();
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
      SgExpression *init_get = BuildGet(
          gd, run_kernel_func, gt, init_indices,
          sil, is_periodic, builder);
      registers[i+1]->reset_initializer(
          sb::buildAssignInitializer(init_get));
      si::fixVariableReferences(registers[i+1]);
    }
    // Insert move statements and a grid get statement for
    // the next point
    if (i < (int)bil.size() - 1) {
      move_stmts.push_back(
          sb::buildAssignStatement(
              sb::buildVarRefExp(reg1),
              sb::buildVarRefExp(registers[i+1])));
    } else {
      // Load a new value to reg
      SgExpressionVector indices_next =
          GetLoopIndices(run_kernel_func, loop);
      OffsetIndices(indices_next, il);
      SgExpression *get_next = BuildGet(
          gd, run_kernel_func, gt, indices_next,
          sil, is_periodic, builder);
      SgExprStatement *asn_stmt =
          sb::buildAssignStatement(sb::buildVarRefExp(reg1),
                                   get_next);
      si::fixVariableReferences(asn_stmt);
      move_stmts.push_back(asn_stmt);
    }
  }
  // If replaced, insert the relevant variable declarations
  // and move statements. If not, delete the ASTs.
  if (replaced) {
    FOREACH (it, registers.begin(), registers.end()) {
      si::insertStatementBefore(loop, *it);
    }
    SgScopeStatement *loop_body =
        isSgScopeStatement(loop->get_loop_body());
    PSAssert(loop_body);
    FOREACH (move_stmts_it, move_stmts.rbegin(),
             move_stmts.rend()) {
      si::prependStatement(*move_stmts_it, loop_body);
      LOG_DEBUG() << "Prepending "
                  << (*move_stmts_it)->unparseToString() << "\n";
    }
  } else {
    FOREACH (move_stmts_it, move_stmts.rbegin(),
             move_stmts.rend()) {
      si::deleteAST(*move_stmts_it);
    }
    FOREACH (it, registers.begin(), registers.end()) {
      si::deleteAST(*it);
    }
  }
  return replaced;
}

//! Apply register blocking to a kernel call.
/*!
  \param proj
  \param tx 
  \param builder
  \param run_kernel_func RunKernel function for the kernel call
  \param loop Stencil loop to apply the optimization
 */
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
  GridMemberRangeMap &gmr = run_kernel_attr->stencil_map()
      ->grid_member_range_map();
  set<SgInitializedName *> gv_set;
  FOREACH (gr_it, gmr.begin(), gmr.end()) {
    SgInitializedName *gv = gr_it->first.first;
    string member = gr_it->first.second;
    const vector<StencilIndexList> &indices = gr_it->second.all_indices();
    vector<IndexListVector> blocked_indices =
        FindCandidateIndices(indices, dim);
    FillStencilHole(blocked_indices, dim);
    FOREACH (blocked_indices_it, blocked_indices.begin(),
             blocked_indices.end()) {
      IndexListVector bil = *blocked_indices_it;
      if (DoRegisterBlockingOneLine(
              tx, builder, GridData(gv, member),
              run_kernel_func, loop, bil)) {
        // Do not apply transformation for non-member access once the
        // grid is applied for member accesses
        gv_set.insert(gv);
      }
    }
  }
  FOREACH (gr_it, gr.begin(), gr.end()) {
    SgInitializedName *gv = gr_it->first;
    if (gv_set.find(gv) != gv_set.end()) continue;
    const vector<StencilIndexList> &indices = gr_it->second.all_indices();
    vector<IndexListVector> blocked_indices =
        FindCandidateIndices(indices, dim);
    FillStencilHole(blocked_indices, dim);
    FOREACH (blocked_indices_it, blocked_indices.begin(),
             blocked_indices.end()) {
      IndexListVector bil = *blocked_indices_it;
      DoRegisterBlockingOneLine(
          tx, builder, GridData(gv), run_kernel_func,
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

