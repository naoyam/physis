// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/map.h"
#include "translator/rose_util.h"
#include "translator/grid.h"
#include "translator/translation_context.h"
#include "translator/physis_names.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

Counter StencilMap::c;

StencilMap::StencilMap(SgFunctionCallExp *call, TranslationContext *tx)
    : id(StencilMap::c.next()) , stencil_type_(NULL),
      func(NULL),
     run_(NULL), run_inner_(NULL), run_boundary_(NULL),
     fc_(call) {
  kernel = StencilMap::getKernelFromMapCall(fc_);
  assert(kernel);
  dom = StencilMap::getDomFromMapCall(fc_);
  assert(dom);
  numDim = tx->findDomain(dom)->num_dims();
  SgExpressionPtrList &args = fc_->get_args()->get_expressions();
  SgInitializedNamePtrList &params = kernel->get_args();
  int param_index = numDim; // skip the index parameters
  // skip the first two args (kernel and domain)
  FOREACH (it, args.begin()+2, args.end()) {
    SgExpression *a = *it;
    if (GridType::isGridType(a->get_type())) {
      SgVarRefExp *gv = isSgVarRefExp(a);
      assert(gv);
      SgInitializedName *n = gv->get_symbol()->get_declaration();
      grid_args_.push_back(n);
      grid_params_.push_back(params[param_index]);
    }
    ++param_index;
  }
  type_ = AnalyzeType(call);
}

StencilMap::Type StencilMap::AnalyzeType(SgFunctionCallExp *call) {
  string map_name = rose_util::getFuncName(call);
  if (map_name == PS_STENCIL_MAP_RB_NAME) {
    return StencilMap::kRedBlack;
  } else if (map_name == PS_STENCIL_MAP_R_NAME) {
    return StencilMap::kRed;
  } else if (map_name == PS_STENCIL_MAP_B_NAME) {
    return StencilMap::kBlack;
  } else {
    return StencilMap::kNormal;
  }
}

SgExpression *StencilMap::getDomFromMapCall(SgFunctionCallExp *call) {
  SgExpressionPtrList &args = call->get_args()->get_expressions();
  SgExpression *domExp = args[1];
  LOG_DEBUG() << "dom: " << domExp->unparseToString() << "\n";
  return domExp;
}

SgFunctionDeclaration *StencilMap::getKernelFromMapCall(
    SgFunctionCallExp *call) {
  SgExpressionPtrList &args = call->get_args()->get_expressions();
  SgExpression *kernelExp = args.front();
  SgFunctionDeclaration *kernel = rose_util::getFuncDeclFromFuncRef(kernelExp);
  LOG_DEBUG() << "kernel: " << kernel->unparseToString() << "\n";
  return kernel;
}


string StencilMap::toString() const {
  ostringstream ss;

  ss << "Call to map with " << rose_util::getName(kernel);

  StringJoin sj(", ");
  FOREACH(git, grid_stencil_range_map().begin(),
          grid_stencil_range_map().end()) {
    git->second.print(sj << (git->first->unparseToString() + ": "));
  }
  ss << ", grid access: " << sj;

  return ss.str();
}

bool StencilMap::isMap(SgFunctionCallExp *call) {
  SgFunctionRefExp *f = isSgFunctionRefExp(call->get_function());
  if (!f) return false;
  SgName name = f->get_symbol()->get_name();
  return name == PS_STENCIL_MAP_NAME ||
      name == PS_STENCIL_MAP_RB_NAME ||
      name == PS_STENCIL_MAP_R_NAME ||
      name == PS_STENCIL_MAP_B_NAME;
}

std::string GridRangeMapToString(GridRangeMap &gr) {
  StringJoin sj;

  FOREACH (it, gr.begin(), gr.end()) {
    sj << it->first->unparseToString()
       << "->" << it->second;
    //sj << (*it->first) << "->" << it->second;
  }
  return "{" + sj.str() + "}";
}

// StencilRange AggregateStencilRange(GridRangeMap &gr,
//                                    const GridSet *gs) {
//   PSAssert(gs->size() > 0);
//   StencilRange stencil_range(gr.find(*(gs->begin()))->second);
//   FOREACH (gsi, gs->begin(), gs->end()) {
//     Grid *g = *gsi;
//     StencilRange &sr = gr.find(g)->second;
//     stencil_range.merge(sr);
//   }
//   return stencil_range;
// }

#if 0
void StencilMap::AnalyzeGridWrites(TranslationContext &tx) {
  
}
#endif


const std::string RunKernelLoopAttribute::name = "RunKernelLoop";
const std::string RunKernelIndexVarAttribute::name = "RunKernelIndexVar";
const std::string RunKernelCallerAttribute::name = "RunKernelCaller";


bool StencilMap::IsGridPeriodic(SgInitializedName *gv) const {
  return isContained<SgInitializedName*>(grid_periodic_set_, gv);
}

void StencilMap::SetGridPeriodic(SgInitializedName *gv) {
  grid_periodic_set_.insert(gv);
}

SgVarRefExp *KernelLoopAnalysis::GetLoopVar(SgForStatement *loop) {
  SgExpression *incr = loop->get_increment();
  SgVarRefExp *v = rose_util::GetUniqueVarRefExp(incr);
  PSAssert(v);
  return v;
}

SgExpression *KernelLoopAnalysis::GetLoopBegin(SgForStatement *loop) {
  const SgStatementPtrList &init_stmt = loop->get_init_stmt();
  SgExpression *begin_exp = NULL;  
  if (init_stmt.size() == 1) {
    SgExprStatement *assign = isSgExprStatement(init_stmt[0]);
    PSAssert(assign);
    begin_exp = isSgAssignOp(
        assign->get_expression())->get_rhs_operand();
  } else if (init_stmt.size() == 0) {
#if 0    
    // backtrack to the previous loop
    SgStatement *preceding_loop = si::getPreviousStatement(loop);
    while (true) {
      if (isSgForStatement(preceding_loop)) break;
      preceding_loop = si::getPreviousStatement(preceding_loop);
    }
    PSAssert(isSgForStatement(preceding_loop));
    begin_exp = KernelLoopAnalysis::GetLoopEnd(
        isSgForStatement(preceding_loop));
#else
    begin_exp = NULL;
#endif
  } else {
    LOG_ERROR() << "Unsupported loop init statement: "
                << loop->unparseToString() << "\n";
    PSAbort(1);
  }
  
  return begin_exp;
}

SgExpression *KernelLoopAnalysis::GetLoopEnd(SgForStatement *loop) {
  SgExprStatement *test = isSgExprStatement(loop->get_test());
  PSAssert(test);
  SgBinaryOp *test_exp = isSgBinaryOp(test->get_expression());
  PSAssert(isSgLessThanOp(test_exp));
  SgExpression *end_exp = test_exp->get_rhs_operand();
  return end_exp;
}

} // namespace translator
} // namespace physis

