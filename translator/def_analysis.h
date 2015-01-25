// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_DEF_ANALYSIS_H_
#define PHYSIS_TRANSLATOR_DEF_ANALYSIS_H_

#include "translator/translator_common.h"
#include "translator/rose_util.h"

namespace physis {
namespace translator {

typedef map<const SgInitializedName*, SgExpressionPtrList> DefMap;

static inline string toString(DefMap &dm) {
  ostringstream ss;
  FOREACH(it, dm.begin(), dm.end()) {
    const SgInitializedName *v = it->first;
    StringJoin sj(",");
    FOREACH(eit, it->second.begin(), it->second.end()) {
      SgExpression *e = *eit;
      if (!e) {
        sj << "NULL";
      } else {
        sj << e->unparseToString();
      }
    }
    ss << v->get_name().getString()
       << " -> {" << sj << "}\n";
  }

  return ss.str();
}

std::auto_ptr<DefMap> findDefinitions(
    SgNode *topLevelNode, const std::vector<SgType*> &relevantTypes);

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_DEF_ANALYSIS_H_ */
