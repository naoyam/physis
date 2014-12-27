// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/map.h"
#include "translator/rose_util.h"
#include "translator/grid.h"
#include "translator/translation_context.h"
#include "translator/physis_names.h"

namespace si = SageInterface;
namespace sb = SageBuilder;
namespace ru = physis::translator::rose_util;

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
  numDim = rose_util::GetASTAttribute<Domain>(dom)->num_dims();
  SgExpressionPtrList &args
      = fc_->get_args()->get_expressions();
  SgExpressionPtrList::iterator args_it = args.begin();
  SgInitializedNamePtrList &params = kernel->get_args();
  int param_index = numDim; // skip the index parameters
  // skip the first two args (kernel and domain)
  args_it += 2;
  // Additionally, skip one for PSStencil param in Fortran
  if (ru::IsFortranLikeLanguage()) ++args_it;
  FOREACH (it, args_it, args.end()) {
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
  SgExpression *domExp = args[ru::IsCLikeLanguage()? 1 : 2];
  LOG_DEBUG() << "dom: " << domExp->unparseToString() << "\n";
  return domExp;
}

SgFunctionDeclaration *StencilMap::getKernelFromMapCall(
    SgFunctionCallExp *call) {
  SgExpressionPtrList &args = call->get_args()->get_expressions();
  SgExpression *kernelExp = args[ru::IsCLikeLanguage()? 0 : 1];
  SgFunctionDeclaration *kernel = rose_util::getFuncDeclFromFuncRef(kernelExp);
  kernel = isSgFunctionDeclaration(kernel->get_definingDeclaration());
  LOG_DEBUG() << "kernel: " << kernel->unparseToString() << "\n";
  return kernel;
}


string StencilMap::toString() const {
  ostringstream ss;

  ss << "Call to map with " << rose_util::getName(kernel);
  return ss.str();
}

bool StencilMap::IsMap(SgFunctionCallExp *call) {
  SgFunctionRefExp *f = isSgFunctionRefExp(call->get_function());
  if (!f) return false;
  SgName name = f->get_symbol()->get_name();
  return
      ((si::is_C_language() || si::is_Cxx_language()) &&
       (name == PS_STENCIL_MAP_NAME ||
        name == PS_STENCIL_MAP_RB_NAME ||
        name == PS_STENCIL_MAP_R_NAME ||
        name == PS_STENCIL_MAP_B_NAME)) ||
      (si::is_Fortran_language() && name == PSF_STENCIL_MAP_NAME);
}

#if 0
void StencilMap::AnalyzeGridWrites(TranslationContext &tx) {
  
}
#endif

string StencilMap::GetTypeName() const {
  return GetInternalNamePrefix() + "PSStencil" + dimStr() +
      "_" + kernel->get_name();
}

string StencilMap::GetMapName() const {
  return GetInternalNamePrefix() + "PSStencil" + dimStr() +
      "Map_" + kernel->get_name();
}

string StencilMap::GetRunName() const {
  return GetInternalNamePrefix() + "PSStencil" + dimStr() +
      "Run_" + kernel->get_name();
}

string StencilMap::GetInternalNamePrefix() {
  if (ru::IsCLikeLanguage()) {
    return string("__");
  } else {
    return string("");
  } 
}


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
  PSAssert(isSgLessOrEqualOp(test_exp) ||
           isSgLessThanOp(test_exp));
  SgExpression *end_exp = test_exp->get_rhs_operand();
  return end_exp;
}

} // namespace translator
} // namespace physis

