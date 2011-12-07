// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/translation_util.h"
#include "translator/rose_util.h"
#include "translator/grid.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

SgType *BuildInt32Type(SgScopeStatement *scope) {
  return sb::buildIntType();
}

SgType *BuildInt64Type(SgScopeStatement *scope) {
  SgType *t = si::lookupNamedTypeInParentScopes("int64_t");
  PSAssert(t);
  return t;
}

SgType *BuildIndexType(SgScopeStatement *scope) {
#if defined(PHYSIS_INDEX_INT32)
  return BuildInt32Type(scope);
#elif defined(PHYSIS_INDEX_INT64)
  return BuildInt64Type(scope);
#else
#error Not supported index type
#endif
}

SgType *BuildIndexType2(SgScopeStatement *scope) {
  return sb::buildOpaqueType(PS_INDEX_TYPE_NAME, scope);
}

SgExpression *BuildIndexVal(index_t v) {
  return rose_util::BuildIntLikeVal(v);
}


SgType *BuildPSOffsetsType() {
  SgType *t =
      si::lookupNamedTypeInParentScopes("__PSOffsets");
  PSAssert(t);
  return t;
}


SgVariableDeclaration *BuildPSOffsets(std::string name,
                                      SgScopeStatement *scope,
                                      __PSOffsets &v) {
  SgType *index_array_type = sb::buildArrayType(BuildIndexType(scope));
  SgExprListExp *elist = sb::buildExprListExp();
  for (int i = 0; i < v.num; ++i) {
    elist->append_expression(rose_util::BuildIntLikeVal(v.offsets[i*2]));
    elist->append_expression(rose_util::BuildIntLikeVal(v.offsets[i*2+1]));
  }
  SgAggregateInitializer *agg_init
      = sb::buildAggregateInitializer(elist);
  SgVariableDeclaration *offsets_decl
      = sb::buildVariableDeclaration(name + "_offsets",
                                     index_array_type,
                                     agg_init,
                                     scope);
  SgAggregateInitializer *psoffset_init
      = sb::buildAggregateInitializer
      (sb::buildExprListExp(rose_util::BuildIntLikeVal(v.num),
                            sb::buildVarRefExp(offsets_decl)));
  SgVariableDeclaration *decl
      = sb::buildVariableDeclaration(name, BuildPSOffsetsType(),
                                     psoffset_init, scope);
  si::appendStatement(decl, scope);
  return decl;
}

SgType *BuildPSGridRangeType() {
  SgType *t =
      si::lookupNamedTypeInParentScopes("__PSGridRange");
  PSAssert(t);
  return t;
}

SgVariableDeclaration *BuildPSGridRange(std::string name,
                                        SgScopeStatement *block,
                                        __PSGridRange &v) {
  SgVariableDeclaration *decl
      = sb::buildVariableDeclaration(name,
                                     BuildPSGridRangeType(),
                                     NULL, block);
  si::appendStatement(decl, block);

  // gr.num_dims = v.num_dims
  SgVarRefExp *decl_ref = sb::buildVarRefExp(decl);
  si::appendStatement(
      sb::buildAssignStatement(
          sb::buildDotExp(decl_ref, sb::buildVarRefExp("num_dims")),
          rose_util::BuildIntLikeVal(v.num_dims)), block);

  for (int i = 0; i < v.num_dims; ++i) {
    // set min_offsets[i]
    SgStatement *stmt;
    SgExpression *min_offsets_exp =
        sb::buildPntrArrRefExp(
            sb::buildDotExp(decl_ref, sb::buildVarRefExp("min_offsets")),
            sb::buildIntVal(i));
    __PSOffsets &min_offsets = v.min_offsets[i];
    stmt =sb::buildAssignStatement(
        sb::buildDotExp(min_offsets_exp, sb::buildVarRefExp("num")),
        rose_util::BuildIntLikeVal(min_offsets.num));
    si::appendStatement(stmt, block);

    // fill __PSOffsets.offsets
    for (int j = 0; j < min_offsets.num; ++j) {
      stmt =sb::buildAssignStatement(
          sb::buildPntrArrRefExp(
              sb::buildDotExp(min_offsets_exp,
                              sb::buildVarRefExp("offsets")),
              sb::buildIntVal(j*2)),
          rose_util::BuildIntLikeVal(min_offsets.offsets[j*2]));
      si::appendStatement(stmt, block);
      stmt =sb::buildAssignStatement(
          sb::buildPntrArrRefExp(
              sb::buildDotExp(min_offsets_exp,
                              sb::buildVarRefExp("offsets")),
              sb::buildIntVal(j*2+1)),
          rose_util::BuildIntLikeVal(min_offsets.offsets[j*2]));
      si::appendStatement(stmt, block);
    }

    SgExpression *max_offsets_exp =
        sb::buildPntrArrRefExp(
            sb::buildDotExp(decl_ref, sb::buildVarRefExp("max_offsets")),
            sb::buildIntVal(i));
    __PSOffsets &max_offsets = v.max_offsets[i];
    stmt =sb::buildAssignStatement(
        sb::buildDotExp(max_offsets_exp, sb::buildVarRefExp("num")),
        rose_util::BuildIntLikeVal(max_offsets.num));
    si::appendStatement(stmt, block);

    for (int j = 0; j < max_offsets.num; ++j) {
      stmt =sb::buildAssignStatement(
          sb::buildPntrArrRefExp(
              sb::buildDotExp(max_offsets_exp,
                              sb::buildVarRefExp("offsets")),
              sb::buildIntVal(j*2)),
          rose_util::BuildIntLikeVal(max_offsets.offsets[j*2]));
      si::appendStatement(stmt, block);
      stmt =sb::buildAssignStatement(
          sb::buildPntrArrRefExp(
              sb::buildDotExp(max_offsets_exp,
                              sb::buildVarRefExp("offsets")),
              sb::buildIntVal(j*2+1)),
          rose_util::BuildIntLikeVal(max_offsets.offsets[j*2]));
      si::appendStatement(stmt, block);
    }
    
  }
  return decl;
}
      
SgExpression *BuildFunctionCall(const std::string &name,
                                SgExpression *arg1) {
  SgFunctionRefExp *function = sb::buildFunctionRefExp(name);
  PSAssert(function);
  SgExprListExp *args = sb::buildExprListExp(arg1);
  PSAssert(args);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(function, args);
  PSAssert(call);
  return call;
}

std::string GetTypeName(SgType *ty) {
  if (isSgTypeFloat(ty)) {
    return string("Float");
  } else if (isSgTypeDouble(ty)) {
    return string("Double");
  } else {
    LOG_ERROR() << "Unsupported type\n";
    PSAbort(1);
    return ""; // just to suppress compiler warning
  }
}

std::string GetTypeDimName(GridType *gt) {
  return GetTypeName(gt->getElmType())
      + toString(gt->getNumDim()) + "D";
}



} // namespace translator
} // namespace physis

