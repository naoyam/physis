// Licensed under the BSD license. See LICENSE.txt for more details.

#include "rose.h"
#include <assert.h>
#include <vector>
#include <boost/foreach.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "translator/ast_traversal.h"
#include "translator/rose_util.h"
#include "translator/test/common.h"

using namespace ::testing;
using namespace ::std;
using namespace ::physis::translator::test;

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace rose_util {


class ast_traversal_FindClosestAncestor: public Test {
 public:
  void SetUp() {
    proj_ = FrontEnd("test_ast_traversal_input.c");
    PSAssert(func_);
  }
  void TearDown() {
    delete proj_;    
  }
  SgProject *proj_;
  SgFunctionDeclaration *func_;
};

TEST_F(ast_traversal_FindClosestAncestor,
       FindsFunctionParameterListForFunctionParameter) {

  SgInitializedName *param = NULL;
  vector<SgInitializedName*> vars =
      si::querySubTree<SgInitializedName>(proj_);
  BOOST_FOREACH(SgInitializedName *v, vars) {
    if (v->get_name() == "param") {
      param = v;
    }
  }
  PSAssert(param);

  SgFunctionParameterList *pl =
      rose_util::FindClosestAncestor<SgFunctionParameterList>(param);

  ASSERT_THAT(pl, NotNull());
}

} // namespace rose_util
} // namespace translator
} // namespace physis
  
int main(int argc, char *argv[]) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}

