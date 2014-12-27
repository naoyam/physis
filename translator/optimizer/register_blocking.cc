// Licensed under the BSD license. See LICENSE.txt for more details.

#include <climits>
#include <algorithm>
#include <boost/foreach.hpp>

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

using std::vector;
using std::string;
using std::map;

using namespace physis;
using namespace physis::translator;

namespace {

typedef vector<StencilRegularIndexList> IndexListVector;

//! Struct to represent a grid variable and a member to access.
struct GridData {
  //SgInitializedName *gv_;
  GridVarAttribute *gva_;
  SgExpression *ref_get_;
  bool member_access_;
  string member_;
  IntVector indices_;
  GridData(GridVarAttribute *gva,
           SgExpression *ref_get,
           const string &member):
      gva_(gva), ref_get_(ref_get), member_access_(true),
      member_(member) {}
  GridData(GridVarAttribute *gva,
           SgExpression *ref_get,
           const string &member, const IntVector &indices):
      gva_(gva), ref_get_(ref_get), member_access_(true),
      member_(member), indices_(indices) {}
  GridData(GridVarAttribute *gva, SgExpression *ref_get):
      gva_(gva), ref_get_(ref_get), member_access_(false) {}
  GridData(const GridData &gd):
      gva_(gd.gva_),
      ref_get_(gd.ref_get_),
      member_access_(gd.member_access_),
      member_(gd.member_), indices_(gd.indices_) {}
  bool IsMemberAccess() const { return member_access_; }
  bool IsArrayMemberAcccess() const {
    return member_access_ && indices_.size() > 0;
  }
};

class RegisterBlocking {
 public:
  SgProject *proj_;
  TranslationContext *tx_;
  RuntimeBuilder *builder_;
  SgFunctionDeclaration *run_kernel_func_;
  SgForStatement *target_loop_;
  RunKernelLoopAttribute *loop_attr_;
  
  RegisterBlocking(SgProject *proj,
                   TranslationContext *tx,
                   RuntimeBuilder *builder,
                   SgFunctionDeclaration *run_kernel_func,
                   SgForStatement *target_loop):
      proj_(proj), tx_(tx), builder_(builder),
      run_kernel_func_(run_kernel_func),
      target_loop_(target_loop) {
    loop_attr_ = rose_util::GetASTAttribute<RunKernelLoopAttribute>(
        target_loop_);
  }
  
  void Run() {
    LOG_DEBUG() << "Checking run kernel function: "
                << run_kernel_func_->get_name() << "\n";
    int dim = loop_attr_->dim();
    vector<SgNode*> target_gets =
        rose_util::QuerySubTreeAttribute<GridGetAttribute>(target_loop_);
    set<GridVarAttribute*> done_grids;
    BOOST_FOREACH (SgNode *get, target_gets) {
      LOG_DEBUG() << "Checking grid get: " << get->unparseToString() << "\n";
      GridGetAttribute *gga = rose_util::GetASTAttribute<GridGetAttribute>(get);
      GridVarAttribute *gva = gga->gva();
      if (isContained(done_grids, gva)) {
        LOG_DEBUG() << "Already processed.\n";
        continue;
      }
      //SgExpression *get_exp = GridGetAnalysis::GetGridExp(isSgExpression(get));
      // If there are member-specific gets, apply register just to
      // member accesses
      bool member_blocked = false;
      LOG_DEBUG() << "Checking member accesses\n";
      BOOST_FOREACH (
          GridVarAttribute::MemberStencilRangeMap::value_type &p,
          gva->member_sr()) {
        const string &member = p.first.first;
        const IntVector &indices = p.first.second;
        const StencilRange &sr = p.second;
        GridData gd(gva, isSgExpression(get), member, indices);      
        member_blocked = 
            ProcessAllIndices(gd, dim, sr) || member_blocked;
      }
      if (!member_blocked) {
        GridData gd(gva, isSgExpression(get));
        ProcessAllIndices(gd, dim, gva->sr());
      }
      done_grids.insert(gva);
    }
  }

  bool ProcessAllIndices(const GridData &gd,
                         int dim,
                         const StencilRange &range) {
    const vector<StencilIndexList> &indices = range.all_indices();
    vector<IndexListVector> blocked_indices =
        FindCandidateIndices(indices, dim);
    FillStencilHole(blocked_indices, dim);
    bool performed = false;
    FOREACH (blocked_indices_it, blocked_indices.begin(),
             blocked_indices.end()) {
      IndexListVector bil = *blocked_indices_it;
      performed = ProcessOneLine(gd, bil) || performed;
    }
    return performed;
  }

  /*!
    Find groups of multiple indices that only differ at the loop
    dimension.  
  */
  vector<IndexListVector> FindCandidateIndices(
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
  

  //! Apply register blocking along one line of accesses.
  /*!
    For example, suppose a stencil:
  
    PSGridGet(x,0,0,-1)+PSGridGet(x,0,0,0)+PSGridGet(x,0,0,1)

    In this case, if the innner most loop is for the last dimension,
    they are replaced with local variable accesses. Parameter bil will
    contain a list, {(0, 0, -1), (0, 0, 0), (0, 0, +1)}, in this case.
  
    \param gd Accessed grid
    \param bil List of target access indices
  */
  bool ProcessOneLine(const GridData &gd, IndexListVector &bil) {
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
    GridType *gt = gd.gva_->gt();
    int dim = loop_attr_->dim();
    // Flag to indicate access replacement 
    bool replaced = false;
    // Local variables
    vector<SgVariableDeclaration*> registers;
    // Declare local variables for register blocking
    for (int i = 0; i < (int)bil.size(); ++i) {
      SgType *ty = GetType(gt, gd);
      // Do not insert the declarations yet since they may not be used.
      SgVariableDeclaration *reg = BuildNewLocalVar(
          ty, si::getEnclosingFunctionDefinition(target_loop_));
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
          ReplaceGetWithReg(target_loop_, gd, &il, reg1);
      // Track if any replacement is done
      replaced = (original_get != NULL) || replaced;
      // There can be no get for this index. The element at that index
      // is still needed to be loaded for blocking.
    
      bool is_periodic = original_get?
          rose_util::GetASTAttribute<GridGetAttribute>(
              original_get)->is_periodic() :
          ShouldGetStencilHolePeriodic(target_loop_, bil, dim, il.GetIndex(dim));
    
      StencilIndexList sil;
      SetStencilIndexList(sil, il);
      // Initial load from memory to registers
      if (i < (int)bil.size() - 1) {
        SgExpressionPtrList init_indices =
            GetLoopIndices(&sil, true);
        //si::deleteAST(init_indices[dim-1]);
        //init_indices[dim-1] = si::copyExpression(loop_attr->begin());
        OffsetIndices(init_indices, il);
        SgExpression *init_get = BuildGet(
            gd, gt, init_indices, sil, is_periodic);
        registers[i+1]->reset_initializer(
            sb::buildAssignInitializer(init_get));
        si::fixVariableReferences(registers[i+1]);
      }
      // Insert move statements and a grid get statement for
      // the next point
      if (i < (int)bil.size() - 1) {
        LOG_DEBUG() << "copying "
                    << registers[i+1]->unparseToString()
                    << " to "
                    << reg1->unparseToString() << "\n";
        move_stmts.push_back(
            sb::buildAssignStatement(
                sb::buildVarRefExp(reg1),
                sb::buildVarRefExp(registers[i+1])));
      } else {
        // Load a new value to reg
        SgExpressionVector indices_next =
            GetLoopIndices(&sil, false);
        OffsetIndices(indices_next, il);
        SgExpression *get_next = BuildGet(
            gd, gt, indices_next, sil, is_periodic);
        LOG_DEBUG() << "Loading "
                    << get_next->unparseToString()
                    << "\n";
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
        si::insertStatementBefore(target_loop_, *it);
      }
      SgScopeStatement *loop_body =
          isSgScopeStatement(
              rose_util::QuerySubTreeAttribute<KernelBody>(
                  target_loop_).front());
      PSAssert(loop_body);
      FOREACH (move_stmts_it, move_stmts.rbegin(),
               move_stmts.rend()) {
        si::prependStatement(*move_stmts_it, loop_body);
        LOG_DEBUG() << "Prepending "
                    << (*move_stmts_it)->unparseToString() << "\n";
      }
    } else {
      LOG_DEBUG() << "No replacement performed.\n";
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

  struct sri_comp {
    int dim;
    bool operator() (const StencilRegularIndexList &x,
                     const StencilRegularIndexList &y) {
      return x.GetIndex(dim) < y.GetIndex(dim);
    }
  };

  // Stencil may have "holes," but they also need to be blocked in
  // registers.
  // Eg., say a kernel has three gets: get(v, x-2), get(v, x), get(v,
  // x+1). When applying register blocking, four registers are required:
  // x-2, x-1, x, and x+1.
  void FillStencilHole(
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

  SgExpression *BuildGet(const GridData &gd,
                         GridType *gt,
                         const SgExpressionPtrList &indices,
                         const StencilIndexList &sil,
                         bool is_periodic) {
    SgExpression *grid_ref = sb::buildVarRefExp(
        GridGetAnalysis::GetGridVar(gd.ref_get_));
    if (gd.IsArrayMemberAcccess()) {
      SgExpressionPtrList array_indices;
      BOOST_FOREACH(int i, gd.indices_) {
        array_indices.push_back(sb::buildIntVal(i));
      }
      return builder_->BuildGridGet(
          grid_ref, gd.gva_, gt, &indices, &sil, true,
          is_periodic, gd.member_, array_indices);
    } else if (gd.IsMemberAccess()) {
      return builder_->BuildGridGet(
          grid_ref, gd.gva_, gt, &indices, &sil, true,
          is_periodic, gd.member_);
    } else {
      return builder_->BuildGridGet(
          grid_ref, gd.gva_, gt, &indices, &sil, true, is_periodic);
    }
  }
  
  SgVariableDeclaration *BuildNewLocalVar(
      SgType *ty, SgScopeStatement *scope) {
    PSAssert(scope);
    string name = rose_util::generateUniqueName(scope);
    //return sb::buildVariableDeclaration(name, ty, NULL, scope);
    return sb::buildVariableDeclaration(name, ty);
  }

  SgExpression *FindGetAtIndex(SgForStatement *loop,
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

  bool DoesMatchForReplacingGetArrayMember(const GridData &gd,
                                           SgExpression *get,
                                           GridGetAttribute *get_attr) {
    if (get_attr->member_name() != gd.member_) return false;
    SgExpressionPtrList array_offset_indices =
        GridOffsetAnalysis::GetArrayOffsetIndices(
            GridGetAnalysis::GetOffset(get));
    IntVector::const_iterator it = gd.indices_.begin();
    int d = 1;
    BOOST_FOREACH(SgExpression *idx_exp, array_offset_indices) {
      int idx;
      if (!rose_util::GetIntLikeVal(idx_exp, idx)) {
        LOG_DEBUG() << "Not static offset: " << idx_exp->unparseToString()
                    << "\n";
        return false;
      }
      if (idx != *it) {
        LOG_DEBUG() << "Different index at dim " << d << ": "
                    << idx << " != " << *it << "\n";
        return false;
      }
      ++d;
      ++it;
    }
    return true;
  }
  
  //!
  /*!
    \param loop
    \param gd
    \param index_list
    \param reg 
    \return one of replaced get expression.
  */
  SgExpression* ReplaceGetWithReg(SgForStatement *loop,
                                  const GridData &gd,
                                  StencilRegularIndexList *index_list,
                                  SgVariableDeclaration *reg) {
    std::vector<SgNode*> gets =
        rose_util::QuerySubTreeAttribute<GridGetAttribute>(loop);
    SgExpression *replaced_get = NULL;
    FOREACH (it, gets.begin(), gets.end()) {
      SgExpression *get = isSgExpression(*it);
      PSAssert(get);
      LOG_DEBUG() << "Replacement target: "
                  << get->unparseToString() << "\n";
      GridGetAttribute *get_attr =
          rose_util::GetASTAttribute<GridGetAttribute>(get);
      if (get_attr->GetStencilIndexList() == NULL) continue;
      if (*index_list != *get_attr->GetStencilIndexList())
        continue;
      GridVarAttribute *gva_base = gd.gva_;
      GridVarAttribute *gva_target = get_attr->gva();
      PSAssert(gva_base);
      PSAssert(gva_target);
      if (gva_base != gva_target) {
        LOG_DEBUG() << "NOT Same grid: "
                    << gd.ref_get_->unparseToString() << ", "
                    << GridGetAnalysis::GetGridVar(get)->get_name() << "\n";
        continue;
      }
      if (isSgVarRefExp(get)) {
        LOG_DEBUG() << "Already replaced\n";
        continue;
      }
      // Ensure the get access is to the same member if its member
      // access
      if (gd.IsArrayMemberAcccess()) {
        if (!DoesMatchForReplacingGetArrayMember(gd, get, get_attr))
          continue;
      } else if (gd.IsMemberAccess()) {
        if (get_attr->member_name() != gd.member_) continue;
      } else {
        // Ignore a get with a member access.
        // This happens when a set of grid accesses have a line of
        // accesses that can be register blocked, but each access is
        // actually a member access. In that case, no register blocking
        // is applied, unless the each member access reads the same
        // member. If the same member is accessed, that case is handled
        // with the true case of this branch (i.e., when
        // gd.member_access_ == true).
        if (get_attr->member_name() != "") continue;
      }
      if (!replaced_get) replaced_get = get;
      SgVarRefExp *reg_ref = sb::buildVarRefExp(reg);
      rose_util::CopyASTAttribute<GridGetAttribute>(reg_ref, get);
      LOG_DEBUG() << "Replacing "
                  << get->unparseToString()
                  << " with " << reg_ref->unparseToString() << "\n";
      si::replaceExpression(get, reg_ref, true);
    }
    return replaced_get;
  }

  static bool index_comp(SgNode *x, SgNode *y) {
    RunKernelIndexVarAttribute *x_attr =
        rose_util::GetASTAttribute<RunKernelIndexVarAttribute>(
            isSgVariableDeclaration(x));
    RunKernelIndexVarAttribute *y_attr =
        rose_util::GetASTAttribute<RunKernelIndexVarAttribute>(
            isSgVariableDeclaration(y));
    return x_attr->dim() < y_attr->dim();
  }
  
  // Set loop var in run_kernel_attr.
  SgExpressionPtrList GetLoopIndices(
      StencilIndexList *sil,
      bool initial_load) {
    SgExpressionPtrList indices;
    std::vector<SgNode*> index_vars = 
        rose_util::QuerySubTreeAttribute<RunKernelIndexVarAttribute>(
            run_kernel_func_);
    std::sort(index_vars.begin(), index_vars.end(), index_comp);
    FOREACH (index_vars_it, index_vars.begin(), index_vars.end()) {
      SgVariableDeclaration *index_var =
          isSgVariableDeclaration(*index_vars_it);
      RunKernelIndexVarAttribute *index_var_attr =
          rose_util::GetASTAttribute<RunKernelIndexVarAttribute>(index_var);
      int idim = index_var_attr->dim();
      bool used = false;
      FOREACH (sil_it, sil->begin(), sil->end()) {
        if (sil_it->dim == idim) {
          used = true;
          break;
        }
      }
      if (used) {
        SgExpression *i = NULL;
        if (initial_load && idim == loop_attr_->dim()) {
          i = KernelLoopAnalysis::GetLoopBegin(target_loop_);
          if (i) {
            i = si::copyExpression(i);
          } else {
            i = sb::buildVarRefExp(index_var);
            rose_util::AddASTAttribute<StencilIndexVarAttribute>(
                i, new StencilIndexVarAttribute(idim));
          }
        } else {
          i = sb::buildVarRefExp(index_var);
          rose_util::AddASTAttribute<StencilIndexVarAttribute>(
              i, new StencilIndexVarAttribute(idim));
        }
        indices.push_back(i);
      }
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

  void OffsetIndices(SgExpressionVector &indices,
                     StencilRegularIndexList &offset) {
    StencilRegularIndexList::map_t::const_iterator sril_it = offset.indices().begin();
    ENUMERATE (i, indices_it, indices.begin(), indices.end()) {
      PSAssert(sril_it != offset.indices().end());
      ssize_t v = sril_it->second;
      if (v != 0) {
        *indices_it = sb::buildAddOp(*indices_it, sb::buildIntVal(v));
      }
      ++sril_it;
    }
    return;
  }

  void ReplaceOffsetIndices(SgFunctionCallExp *get_offset,
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

  void ReplaceGetIndices(SgExpression *get,
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

  SgExpression *ReplaceGridInGet(SgExpression *original_get,
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

  bool ShouldGetStencilHolePeriodic(SgForStatement *loop,
                                    IndexListVector &bil,
                                    int dim, int index) {
    // Whether this index should be accessed with the periodic
    // boundary condition can't be determined if this index is
    // originally a stencil hole. Conservatively determine whether it
    // should be periodic or not.
    if (index == 0) return false;
    FOREACH (bil_it, bil.begin(), bil.end()) {
      StencilRegularIndexList &il = *bil_it;
      if (!((index > 0 && il.GetIndex(dim) > 0) ||
            (index < 0 && il.GetIndex(dim) < 0))) {
        continue;
      }
      SgExpression *get = FindGetAtIndex(loop, &il);
      // no get for a stencil hole
      if (!get) continue;
      bool p = rose_util::GetASTAttribute<GridGetAttribute>(get)->is_periodic();
      if (p) return true;
    }
    return false;    
  }


  void SetStencilIndexList(StencilIndexList &sil,
                           const StencilRegularIndexList &sril) {
    const StencilRegularIndexList::map_t indices = sril.indices();
    FOREACH (it, indices.begin(), indices.end()) {
      sil.push_back(StencilIndex(it->first, it->second));
    }
  }


  //! Get type of a grid data.
  /*!
    If it's an access to member, returns its member type; otherwise, the
    grid point type, which may be a struct type.
  
    \param gt
    \param gd
  */
  SgType *GetType(GridType *gt, const GridData &gd) {
    if (gt->IsPrimitivePointType() || !gd.IsMemberAccess()) {
      return gt->point_type();
    } else {
      SgClassDefinition *utdef = gt->point_def();
      vector<SgVariableDeclaration*> mdecls =
          si::querySubTree<SgVariableDeclaration>(
              utdef, V_SgVariableDeclaration);
      FOREACH (it, mdecls.begin(), mdecls.end()) {
        SgVariableDeclaration *mdecl = *it;
        if (rose_util::GetName(mdecl) == gd.member_) {
          SgType *ty = mdecl->get_definition()->get_type();
          if (gd.indices_.size() > 0) {
            PSAssert(isSgArrayType(ty));
            PSAssert(si::getDimensionCount(ty) == (int)gd.indices_.size());
            ty = si::getArrayElementType(ty);
          }
          return ty;
        }
      }
      LOG_ERROR() << "No such member found: " << gd.member_
                  << "\n";
      return NULL;
    }
  }
  
};

} // namespace

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

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
    RunKernelAttribute *run_kernel_attr =
        rose_util::GetASTAttribute<RunKernelAttribute>(run_kernel_func);
    // Do not apply the optimization if this loop is for the red-black ordering
    if (run_kernel_attr->stencil_map()->IsRedBlackVariant()) {
      LOG_DEBUG() << "Register blocking not applied for stencils with the red-black ordering\n";
      continue;
    }
#if 0    
    DoRegisterBlocking(proj, tx, builder, run_kernel_func,
                       target_loop);
#else
    RegisterBlocking rb(proj, tx, builder, run_kernel_func,
                        target_loop);
    rb.Run();
#endif
  }

  post_process(proj, tx, __FUNCTION__);
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

