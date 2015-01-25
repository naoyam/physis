// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_RUN_H_
#define PHYSIS_TRANSLATOR_RUN_H_

#include <vector>

#include "translator/translator_common.h"
#include "physis/physis_util.h"
#include "translator/map.h"

namespace physis {
namespace translator {

class TranslationContext;

class Run {
  SgFunctionCallExp *call;
  SgExpression *count_;
  typedef std::vector<std::pair<SgExpression*, StencilMap*> >
  StencilMapArgVector;
  StencilMapArgVector stencils_;
 public:
  Run(SgFunctionCallExp *call, TranslationContext *tx);
  virtual ~Run() {}

  string GetName() const {
    return "__" + string(PS_STENCIL_RUN_NAME) + "_" + toString(id_);
  }

  const StencilMapArgVector &stencils() const { return stencils_; }
  bool HasCount() const;
  SgExpression *BuildCount() const;

  static bool isRun(SgFunctionCallExp *call);
  static SgExpression *findCountArg(SgFunctionCallExp *call);
#ifdef UNUSED_CODE
  virtual bool IsRead(Grid *g, TranslationContext *tx);
  virtual bool IsReadAny(GridSet *gs, TranslationContext *tx);
  virtual bool IsModified(Grid *g, TranslationContext *tx);
  virtual bool IsModifiedAny(GridSet *gs, TranslationContext *tx);
#endif
  
  int id() const { return id_; }
  
 protected:
  int id_;
  static Counter c;
};

} // namespace translator
} // namespace physis

#endif /* RUN_H_ */
