// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/builder_interface.h"
#include "translator/translation_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

void primitive_optimization(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder) {
  pre_process(proj, tx, __FUNCTION__);
  
  vector<SgForStatement*> target_loops = FindInnermostLoops(proj);
  FOREACH (it, target_loops.begin(), target_loops.end()) {
    SgForStatement *loop = *it;
    EliminateDeadCode(loop);
  }
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

