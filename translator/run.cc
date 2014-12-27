// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/run.h"
#include "translator/rose_util.h"
#include "translator/translation_context.h"

namespace si = SageInterface;
namespace sb = SageBuilder;
namespace ru = physis::translator::rose_util;

namespace physis {
namespace translator {

Counter Run::c;

Run::Run(SgFunctionCallExp *call, TranslationContext *tx)
    : call(call), id_(Run::c.next()) {
  count_ = Run::findCountArg(call);
  SgExpressionPtrList::iterator begin, end;
  if (ru::IsCLikeLanguage()) {
    SgExpressionPtrList &args =
        call->get_args()->get_expressions();
    begin = args.begin();
    end = args.end();
    if (count_) --end;
  } else if (ru::IsFortranLikeLanguage()) {
    SgAggregateInitializer *smap_expr =
        isSgAggregateInitializer(call->get_args()->get_expressions()[0]);
    PSAssert(smap_expr);
    LOG_DEBUG() << "Smap arg: " << smap_expr->unparseToString() << "\n";
    SgExpressionPtrList &smap_args =
        isSgExprListExp(smap_expr->get_initializers()->get_expressions()[0])
        ->get_expressions();
    begin = smap_args.begin();
    end = smap_args.end();
  }
    
  FOREACH(it, begin, end) {
    LOG_DEBUG() << "stencil_run stencil arg: "
                << (*it)->unparseToString() << "\n";
    StencilMap *s = tx->findMap(*it);
    assert(s);
    stencils_.push_back(std::make_pair(*it, s));
  }
}

bool Run::isRun(SgFunctionCallExp *call) {
  SgFunctionRefExp *f = isSgFunctionRefExp(call->get_function());
  if (!f) return false;
  SgName name = f->get_symbol()->get_name();
  return name == PS_STENCIL_RUN_NAME;
}

SgExpression *Run::findCountArg(SgFunctionCallExp *call) {
  SgExpressionPtrList &args =
      call->get_args()->get_expressions();
  SgExpression *lastArg = args.back();
  LOG_DEBUG() << "Last arg: " << lastArg->unparseToString() << "\n";
  SgType *lastArgType = lastArg->get_type();
  if (lastArgType->isIntegerType()) {
    LOG_DEBUG() << "last arg is integer\n";
    return lastArg;
  } else {
    return NULL;
  }
}

#ifdef UNUSED_CODE
bool Run::IsRead(Grid *g, TranslationContext *tx) {
  FOREACH (sit, stencils_.begin(), stencils_.end()) {
    StencilMap *smap = sit->second;
    Kernel *kernel = tx->findKernel(smap->getKernel());
    PSAssert(kernel);
    if (kernel->isRead(g)) return true;
  }
  return false;
}

bool Run::IsReadAny(GridSet *gs, TranslationContext *tx) {
  FOREACH (sit, stencils_.begin(), stencils_.end()) {
    StencilMap *smap = sit->second;
    Kernel *kernel = tx->findKernel(smap->getKernel());
    PSAssert(kernel);
    if (kernel->isReadAny(gs)) return true;
  }
  return false;
}

bool Run::IsModified(Grid *g, TranslationContext *tx) {
  FOREACH (sit, stencils_.begin(), stencils_.end()) {
    StencilMap *smap = sit->second;
    Kernel *kernel = tx->findKernel(smap->getKernel());
    PSAssert(kernel);
    if (kernel->isModified(g)) return true;
  }
  return false;
}

bool Run::IsModifiedAny(GridSet *gs, TranslationContext *tx) {
  FOREACH (sit, stencils_.begin(), stencils_.end()) {
    StencilMap *smap = sit->second;
    Kernel *kernel = tx->findKernel(smap->getKernel());
    PSAssert(kernel);
    if (kernel->isModifiedAny(gs)) return true;
  }
  return false;
}
#endif

bool Run::HasCount() const {
  return count_ != NULL;
}

SgExpression *Run::BuildCount() const {
  return (count_) ? si::copyExpression(count_) : NULL;
}

} // namespace translator
} // namespace physis

