// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_STENCIL_ANALYSIS_H_
#define PHYSIS_TRANSLATOR_STENCIL_ANALYSIS_H_

#include "translator/translator_common.h"
#include "translator/map.h"

namespace physis {
namespace translator {

bool AnalyzeStencilIndex(SgExpression *arg, StencilIndex &idx,
                         SgFunctionDeclaration *kernel);
void AnalyzeStencilRange(StencilMap &sm, TranslationContext &tx);

//void AnalyzeEmit(SgFunctionDeclaration *func);

void AnalyzeGet(SgNode *top_level_node,
                TranslationContext &tx);
void AnalyzeEmit(SgNode *top_level_node,
                 TranslationContext &tx);

/*!
  
  \param get
  \param indices
  \param parent
  \return True upon success; fasle otherwise.
*/
bool AnalyzeGetArrayMember(SgDotExp *get, SgExpressionVector &indices,
                           SgExpression *&parent);


} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_STENCIL_ANALYSIS_H_ */
