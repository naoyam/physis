// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/translator_common.h"
#include "translator/translation_context.h"
#include "translator/builder_interface.h"

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

inline void pre_process(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    const char *pass_name) {
  LOG_INFO() << "OPT: " << pass_name << "\n";
}

inline void post_process(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    const char *pass_name) {
  // Validate AST
  if (!getenv("NOASTCHECK")) {
    LOG_DEBUG() << "Validating AST\n";    
    AstTests::runAllTests(proj);
  }
}


//! Do-nothing optimization pass.
/*!
  @proj The whole AST.
  @param tx The translation context built for the AST.
 */
extern void null_optimization(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

//! Apply primitive optimizations
/*!
  @proj The whole AST.
  @param tx The translation context built for the AST.
 */
extern void primitive_optimization(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

//! Inline stencil kernels.
/*!
  This optimization itself may have no performance improvement, but
  is required by other optimizations such as register blocking.
  
  From:
  \code
  void kernel(const int x, const int y, ...) {
  }
  void run_kernel(...) {
    for {
      kernel(...);
    }
  }
  \endcode

  To:
  \code
  void kernel(const int x, const int y, ...) {
  }
  void run_kernel(...) {
    for {
      <kernel inlined here>
    }
  }
  \endcode
*/
extern void kernel_inlining(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

extern void loop_peeling(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

extern void register_blocking(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

//! Common subexpression elimination in grid offset calculations.
/*!
  From:
  \code
  g[x + y*nx + z*nx*ny] // GridGet(x, y, z);
  g[x+1 + y*nx + z*nx*ny] // GridGet(x+1, y, z);  
  \endcode

  To:
  \code
  int index = x + y*nx + z*nx*ny;
  g[index] // GridGet(x, y, z);
  g[index+1] // GridGet(x+1, y, z);
  \endcode
 */
extern void offset_cse(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);
    
//! Make conditional get unconditional.
/*!
  Assumes grid_index_cse is already applied.

  From:
  \code
  int base_index = ;
  if (cond) {
    f = g[base_index+a];
  }
  \endcode

  To:
  \code
  int base_index = ;
  int findex = base_index;
  if (cond) {
    findex += a;
  }
  f = g[findex]
  \endcode
  
 */
extern void unconditional_get(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);


//! Apply CSE to offset calculation.
/*!
 */
extern void offset_cse(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

//! Apply CSE to offset calculation across loop iterations.
/*!
 */
extern void offset_spatial_cse(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

//! Miscellaneous loop optimizations
/*!
 */
extern void loop_opt(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder);

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis
