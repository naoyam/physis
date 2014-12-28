// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/optimization_passes.h"
#include "translator/rose_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

void null_optimization(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder) {
  pre_process(proj, tx, __FUNCTION__);
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

