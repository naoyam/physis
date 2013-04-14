// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/stencil_analysis.h"
#include "translator/translation_context.h"

using namespace std;
namespace si = SageInterface;

namespace physis {
namespace translator {
#if 0
    FOREACH(gsi, gs->begin(), gs->end()) {
      Grid *g = *gsi;
      g->access(offset);
      StencilRange &sr = uf.gr[g];
      LOG_DEBUG() << "Current range: " << sr.toString() << "\n";
      sr.insert(offset);
      LOG_DEBUG() << "After insertion:: " << sr.toString() << "\n";
    }
#endif

static bool getIntValue(SgExpression *exp, ssize_t &ret) {
  if (isSgIntVal(exp)) {
    ret = isSgIntVal(exp)->get_value();
    return true;
  } else if (isSgLongIntVal(exp)) {
    ret = isSgLongIntVal(exp)->get_value();    
    return true;
  } else if (isSgLongLongIntVal(exp)) {
    ret = isSgLongLongIntVal(exp)->get_value();    
    return true;
  } else if (isSgUnsignedIntVal(exp)) {
    ret = isSgUnsignedIntVal(exp)->get_value();    
    return true;
  } else {
    return false;
  }
}

static bool AnalyzeVarRef(SgVarRefExp *vref,
                          SgFunctionDeclaration *kernel,
                          int &dim) {
  SgInitializedNamePtrList &args = kernel->get_args();
  SgInitializedName *vn = vref->get_symbol()->get_declaration();
  ENUMERATE (i, ait, args.begin(), args.end()) {
    if (*ait == vn) {
      // NOTE: dimenstion starts at 1 (dimension value of 0 is
      // reserved for the cases where no index var is used)
      dim = i+1;
      LOG_DEBUG() << "Reference to index " << dim << " found\n";
      return true;
    }
  }
  return false;
}

// Legitimate expression:
// - (integer constant) * (index variable) + integer constant
bool AnalyzeStencilIndex(SgExpression *arg, StencilIndex &idx,
                         SgFunctionDeclaration *kernel) {
  ssize_t v;
  if (getIntValue(arg, v)) {
    idx.offset = v;
    idx.dim = 0;
    LOG_DEBUG() << "Integer constant: " << v << "\n";
    return true;
  }

  // var ref
  if (isSgVarRefExp(arg)) {
    int dim;
    if (AnalyzeVarRef(isSgVarRefExp(arg), kernel, dim)) {
      idx.dim = dim;
      idx.offset = 0;
      LOG_DEBUG() << "Index variable reference\n";
      return true;
    } else {
      LOG_DEBUG() << "Invalid variable reference\n";
    }
  }
    
  if (isSgAddOp(arg) || isSgSubtractOp(arg)) {
    SgBinaryOp *bop = isSgBinaryOp(arg);
    SgExpression *rhs = bop->get_rhs_operand();
    SgExpression *lhs = bop->get_lhs_operand();
    StencilIndex rhs_index;
    StencilIndex lhs_index;
    AnalyzeStencilIndex(rhs, rhs_index, kernel);
    AnalyzeStencilIndex(lhs, lhs_index, kernel);
    if (rhs_index.dim != 0 && lhs_index.dim != 0) {
      LOG_DEBUG() << "Invalid: two variable use\n";
      PSAbort(1);
    } else if (rhs_index.dim == 0 && lhs_index.dim == 0) {
      LOG_ERROR() << "Invalid: LHS and RHS are both constant, but this should not happen because constant folding is applied before this routine.\n";
      PSAbort(1);
    }

    // Make the variable reference lhs
    if (rhs_index.dim != 0) {
      std::swap(rhs_index, lhs_index);
    }
    
    PSAssert(rhs_index.dim == 0);
    idx.dim = lhs_index.dim;
    idx.offset = rhs_index.offset;
    if (isSgSubtractOp(arg)) {
      idx.offset *= -1;
    }
    // This should be always true if constant folding works beyond brace
    // boundaries, but apparently it's not the case.
    // PSAssert(lhs_index.offset == 0);
    if (lhs_index.offset != 0) {
      idx.offset += lhs_index.offset;
    }
    return true;
  }

  if (isSgCastExp(arg)) {
    SgCastExp *ca = isSgCastExp(arg);
    return AnalyzeStencilIndex(ca->get_operand(), idx, kernel);
  }

  if (isSgMinusOp(arg)) {
    if (AnalyzeStencilIndex(isSgMinusOp(arg)->get_operand(), idx,
                            kernel)) {
      idx.dim *= -1;
      idx.offset *= -1;
      return true;
    }
  }

  LOG_WARNING() << "Invalid stencil index: "
                << arg->unparseToString() << " ("
                << arg->class_name() << ")\n";
  return false;
}

static void PropagateStencilRangeToGrid(StencilMap &sm, TranslationContext &tx) {
  GridRangeMap &gr = sm.grid_stencil_range_map();
  FOREACH (it, gr.begin(), gr.end()) {
    SgInitializedName *gn = it->first;
    StencilRange &sr = it->second;
    const GridSet *gs = tx.findGrid(gn);
    FOREACH (git, gs->begin(), gs->end()) {
      Grid *g = *git;
      // Note: g is NULL if InitializedName can be initialized
      // with external variables. That happens if a stencil kernel is
      // not designated as static.
      if (g == NULL) {
        LOG_INFO() << "Externally passed grid not set with stencil range info.\n";
        continue;
      } 
      g->SetStencilRange(sr);
      LOG_DEBUG() << "Grid stencil range: "
                  << *g << ", " << sr << "\n";
    }
  }
}

static bool GetGridMember(SgFunctionCallExp *get_call,
                          string &member) {
  SgDotExp *dot = isSgDotExp(get_call->get_parent());
  if (dot == NULL) return false;
  SgVarRefExp *rhs = isSgVarRefExp(dot->get_rhs_operand());
  PSAssert(rhs);
  member = rose_util::GetName(rhs);
  return true;
}

static void AddIndex(SgFunctionCallExp *get_call,
                     GridRangeMap &gr,
                     GridMemberRangeMap &gmr,
                     SgInitializedName *gv,
                     StencilIndexList &sil, int nd) {
  string member;
  if (GetGridMember(get_call, member)) {
    LOG_DEBUG() << "Access to member: " << member << "\n";
    if (!isContained<GridMember, StencilRange>(gmr, GridMember(gv, member))) {
      gmr.insert(make_pair(GridMember(gv, member), StencilRange(nd)));
    }
    StencilRange &sr = gmr.find(GridMember(gv, member))->second;
    sr.insert(sil);
  }    
  if (!isContained<SgInitializedName*, StencilRange>(gr, gv)) {
    gr.insert(make_pair(gv, StencilRange(nd)));
  }
  StencilRange &sr = gr.find(gv)->second;
  sr.insert(sil);
  return;
}

void AnalyzeStencilRange(StencilMap &sm, TranslationContext &tx) {
  SgFunctionDeclaration *kernel = sm.getKernel();
  SgFunctionCallExpPtrList get_calls
      = tx.getGridGetCalls(kernel->get_definition());
  SgFunctionCallExpPtrList get_periodic_calls
      = tx.getGridGetPeriodicCalls(kernel->get_definition());
  get_calls.insert(get_calls.end(), get_periodic_calls.begin(),
                   get_periodic_calls.end());
  GridRangeMap &gr = sm.grid_stencil_range_map();
  GridMemberRangeMap &gmr = sm.grid_member_range_map();  
  FOREACH (it, get_calls.begin(), get_calls.end()) {
    SgFunctionCallExp *get_call = *it;
    LOG_DEBUG() << "Get call detected: "
                << get_call->unparseToString() << "\n";
    SgInitializedName *gv = GridType::getGridVarUsedInFuncCall(get_call);
    SgExpressionPtrList &args = get_call->get_args()->get_expressions();
    int nd = args.size();
    StencilIndexList stencil_indices;
    ENUMERATE (dim, ait, args.begin(), args.end()) {
      LOG_DEBUG() << "get argument: " <<
          dim << ", " << (*ait)->unparseToString() << "\n";
      StencilIndex si;
      SgExpression *arg = *ait;
      // Simplify the analysis
      LOG_DEBUG() << "Analyzing " << arg->unparseToString() << "\n";
      si::constantFolding(arg->get_parent());
      LOG_VERBOSE() << "Constant folded: " << arg->unparseToString() << "\n";
      
      if (!AnalyzeStencilIndex(arg, si, kernel)) {
        PSAbort(1);
      }
      stencil_indices.push_back(si);
    }
    AddIndex(get_call, gr, gmr, gv, stencil_indices, nd);    
    tx.registerStencilIndex(get_call, stencil_indices);
    LOG_DEBUG() << "Analyzed index: " << stencil_indices << "\n";
    bool is_periodic = tx.getGridFuncName(get_call) ==
        GridType::get_periodic_name;
    // A kernel function is analyzed multiple times if it appears
    // multiple times in use at stencil_map.
    if (rose_util::GetASTAttribute<GridGetAttribute>(
            get_call) == NULL) {
      GridGetAttribute *gga = new GridGetAttribute(
          gv, nd, tx.isKernel(kernel),
          is_periodic, &stencil_indices);
      rose_util::AddASTAttribute(get_call, gga);
    }
    if (is_periodic) sm.SetGridPeriodic(gv);
  }
  LOG_DEBUG() << "Stencil access: "
              << GridRangeMapToString(gr)
              << std::endl;
  PropagateStencilRangeToGrid(sm, tx);
}

} // namespace translator
} // namespace physis
