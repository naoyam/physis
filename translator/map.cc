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

namespace physis {
namespace translator {

Counter StencilMap::c;

StencilMap::StencilMap(SgFunctionCallExp *call, TranslationContext *tx)
    :id(StencilMap::c.next()) , stencil_type_(NULL), func(NULL),
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
  return name == MAP_NAME;
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

} // namespace translator
} // namespace physis

