// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_TRANSLATOR_UTIL_H_
#define PHYSIS_TRANSLATOR_TRANSLATOR_UTIL_H_

#include "physis/physis_common.h"
#include "translator/translator_common.h"
#include "translator/grid.h"
namespace physis {
namespace translator {

SgType *BuildInt32Type(SgScopeStatement *scope=NULL);
SgType *BuildInt64Type(SgScopeStatement *scope=NULL);
SgType *BuildIndexType(SgScopeStatement *scope=NULL);
SgType *BuildIndexType2(SgScopeStatement *scope=NULL);

SgExpression *BuildIndexVal(PSIndex v);

SgType *BuildPSOffsetsType();
SgVariableDeclaration *BuildPSOffsets(std::string name,
                                      SgScopeStatement *scope,
                                      __PSOffsets &v);

SgType *BuildPSGridRangeType();
SgVariableDeclaration *BuildPSGridRange(std::string name,
                                        SgScopeStatement *block,
                                        __PSGridRange &v);

SgExpression *BuildFunctionCall(const std::string &name,
                                SgExpression *arg1);

std::string GetTypeName(SgType *ty);
std::string GetTypeDimName(GridType *gt);
  
SgType *GetBaseType(SgType *ty);
  
} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_TRANSLATOR_UTIL_H_ */
