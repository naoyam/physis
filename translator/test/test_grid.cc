#include "rose.h"
#include <assert.h>
#include <vector>
#include <boost/foreach.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "translator/test/common.h"
#include "translator/ast_processing.h"
#include "translator/rose_util.h"
#include "translator/grid.h"

using namespace ::testing;
using namespace ::std;
using namespace ::physis::translator::test;

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

class GridOffsetAnalysis_GetGridVar: public Test {
 public:
  void SetUp() {
    proj_ = FrontEnd("test_grid_input.c");
  }
  void TearDown() {
    delete proj_;
  }
  SgProject *proj_;
};

TEST_F(GridOffsetAnalysis_GetGridVar,
       ReturnsFirstVarRefExpInOffsetCall) {
  SgScopeStatement *scope = si::getFirstGlobalScope(proj_);
  SgVariableDeclaration *x = sb::buildVariableDeclaration(
      "x", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *i = sb::buildVariableDeclaration(
      "i", sb::buildIntType(), NULL, scope);
  SgFunctionCallExp *offset =
      sb::buildFunctionCallExp(
          "offset",
          sb::buildVoidType(),
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(x)),
              sb::buildVarRefExp(i)), scope);
  
  SgVarRefExp *v = GridOffsetAnalysis::GetGridVar(offset);
  ASSERT_THAT(v->get_symbol()->get_declaration()->get_declaration(), Eq(x));
}

TEST_F(GridOffsetAnalysis_GetGridVar,
       ReturnsFirstVarRefExpInUTypeOffsetExp) {
  SgScopeStatement *scope = si::getFirstGlobalScope(proj_);
  SgVariableDeclaration *x = sb::buildVariableDeclaration(
      "x", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *i = sb::buildVariableDeclaration(
      "i", sb::buildIntType(), NULL, scope);
  SgFunctionCallExp *offset =
      sb::buildFunctionCallExp(
          "offset",
          sb::buildVoidType(),
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(x)),
              sb::buildVarRefExp(i)), scope);
  SgExpression *offset_utype =
      sb::buildAddOp(offset,
                     sb::buildVarRefExp(i));
  
  SgVarRefExp *v = GridOffsetAnalysis::GetGridVar(offset_utype);
  ASSERT_THAT(v, NotNull());
  ASSERT_THAT(v->get_symbol()->get_declaration()->get_declaration(), Eq(x));
}

class GridOffsetAnalysis_GetIndexAt: public Test {
 public:
  void SetUp() {
    proj_ = FrontEnd("test_grid_input.c");
  }
  void TearDown() {
    delete proj_;
  }
  SgProject *proj_;
};

TEST_F(GridOffsetAnalysis_GetIndexAt,
       ReturnsIndexForFirstDimWhenGivenOne) {
  SgScopeStatement *scope = si::getFirstGlobalScope(proj_);
  SgVariableDeclaration *x = sb::buildVariableDeclaration(
      "x", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *i = sb::buildVariableDeclaration(
      "i", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *j = sb::buildVariableDeclaration(
      "j", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *k = sb::buildVariableDeclaration(
      "k", sb::buildIntType(), NULL, scope);
  SgFunctionCallExp *offset =
      sb::buildFunctionCallExp(
          "offset",
          sb::buildVoidType(),
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(x)),
              sb::buildVarRefExp(i),
              sb::buildVarRefExp(j),
              sb::buildVarRefExp(k)),
          scope);
  SgExpression *offset_utype =
      sb::buildAddOp(offset,
                     sb::buildVarRefExp(i));
  
  SgExpression *index = GridOffsetAnalysis::GetIndexAt(offset_utype, 1);
  ASSERT_THAT(index, NotNull());
  ASSERT_TRUE(isSgVarRefExp(index));
  ASSERT_THAT(isSgVarRefExp(index)->get_symbol()->get_declaration()->get_declaration(),
              Eq(i));
}

class GridOffsetAnalysis_GetIndices: public Test {
 public:
  void SetUp() {
    proj_ = FrontEnd("test_grid_input.c");
  }
  void TearDown() {
    delete proj_;
  }
  SgProject *proj_;
};

TEST_F(GridOffsetAnalysis_GetIndices,
       ReturnsAllIndexExpressions) {
  SgScopeStatement *scope = si::getFirstGlobalScope(proj_);
  SgVariableDeclaration *x = sb::buildVariableDeclaration(
      "x", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *i = sb::buildVariableDeclaration(
      "i", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *j = sb::buildVariableDeclaration(
      "j", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *k = sb::buildVariableDeclaration(
      "k", sb::buildIntType(), NULL, scope);
  SgFunctionCallExp *offset =
      sb::buildFunctionCallExp(
          "offset",
          sb::buildVoidType(),
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(x)),
              sb::buildVarRefExp(i),
              sb::buildVarRefExp(j),
              sb::buildVarRefExp(k)),
          scope);
  SgExpression *offset_utype =
      sb::buildAddOp(offset,
                     sb::buildVarRefExp(i));
  
  SgExpressionPtrList indices = GridOffsetAnalysis::GetIndices(offset_utype);
  ASSERT_THAT(indices.size(), Eq(3U));
  ASSERT_THAT(isSgVarRefExp(indices[0])->get_symbol()->get_declaration()->get_declaration(),
              Eq(i));
  ASSERT_THAT(isSgVarRefExp(indices[1])->get_symbol()->get_declaration()->get_declaration(),
              Eq(j));
  ASSERT_THAT(isSgVarRefExp(indices[2])->get_symbol()->get_declaration()->get_declaration(),
              Eq(k));
}

class GridGetAnalysis_GetOffset: public Test {
 public:
  void SetUp() {
    proj_ = FrontEnd("test_grid_input.c");
  }
  void TearDown() {
    delete proj_;
  }
  SgProject *proj_;
};

TEST_F(GridGetAnalysis_GetOffset,
       ReturnsOffsetExpression) {
  SgScopeStatement *scope = si::getFirstGlobalScope(proj_);
  SgVariableDeclaration *x = sb::buildVariableDeclaration(
      "x", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *i = sb::buildVariableDeclaration(
      "i", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *j = sb::buildVariableDeclaration(
      "j", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *k = sb::buildVariableDeclaration(
      "k", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *g = sb::buildVariableDeclaration(
      "g", sb::buildPointerType(sb::buildIntType()), NULL, scope);
  
  SgFunctionCallExp *offset =
      sb::buildFunctionCallExp(
          "offset",
          sb::buildVoidType(),
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(x)),
              sb::buildVarRefExp(i),
              sb::buildVarRefExp(j),
              sb::buildVarRefExp(k)),
          scope);
  SgExpression *offset_utype =
      sb::buildAddOp(offset,
                     sb::buildVarRefExp(i));
  SgExpression *get =
      sb::buildPntrArrRefExp(sb::buildVarRefExp(g),
                             offset_utype);
  
  SgExpression *o = GridGetAnalysis::GetOffset(get);
  ASSERT_THAT(o, NotNull());
  ASSERT_THAT(o, Eq(offset_utype));
}

class GridGetAnalysis_GetGridVar: public Test {
 public:
  void SetUp() {
    proj_ = FrontEnd("test_grid_input.c");
  }
  void TearDown() {
    delete proj_;
  }
  SgProject *proj_;
};

TEST_F(GridGetAnalysis_GetGridVar,
       ReturnsVariableUsedAsGrid) {
  SgScopeStatement *scope = si::getFirstGlobalScope(proj_);
  SgVariableDeclaration *x = sb::buildVariableDeclaration(
      "x", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *i = sb::buildVariableDeclaration(
      "i", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *j = sb::buildVariableDeclaration(
      "j", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *k = sb::buildVariableDeclaration(
      "k", sb::buildIntType(), NULL, scope);
  SgVariableDeclaration *g = sb::buildVariableDeclaration(
      "g", sb::buildPointerType(sb::buildIntType()), NULL, scope);
  SgVariableDeclaration *p = sb::buildVariableDeclaration(
      "g", sb::buildPointerType(sb::buildIntType()), NULL, scope);
  
  SgFunctionCallExp *offset =
      sb::buildFunctionCallExp(
          "offset",
          sb::buildVoidType(),
          sb::buildExprListExp(
              sb::buildAddressOfOp(sb::buildVarRefExp(x)),
              sb::buildVarRefExp(i),
              sb::buildVarRefExp(j),
              sb::buildVarRefExp(k)),
          scope);
  SgExpression *offset_utype =
      sb::buildAddOp(offset,
                     sb::buildVarRefExp(i));
  SgExpression *get =
      sb::buildPntrArrRefExp(
          sb::buildArrowExp(sb::buildVarRefExp(g),
                            sb::buildVarRefExp(p)),
          offset_utype);
  
  SgInitializedName *o = GridGetAnalysis::GetGridVar(get);
  ASSERT_THAT(o, NotNull());
  ASSERT_THAT(o, Eq(g->get_variables()[0]));
}


} // namespace translator
} // namespace physis

  
int main(int argc, char *argv[]) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
