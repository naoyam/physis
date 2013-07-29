#include "translator/test/common.h"

#include <vector>

using namespace ::std;

namespace physis {
namespace translator {
namespace test {

SgProject *FrontEnd(const char *infile) {
  vector<string> argv;
  argv.push_back("test");
  argv.push_back(infile);
  SgProject* proj = frontend(argv);
  AstTests::runAllTests(proj);
  return proj;
}


} // namespace test
} // namespace translator
} // namespace physis
