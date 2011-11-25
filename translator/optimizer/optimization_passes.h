// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/translator_common.h"
#include "translator/translation_context.h"

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


//! Do-nothing optimization pass.
/*!
  @proj The whole AST.
  @param tx The translation context built for the AST.
 */
extern void null_optimization(
    SgProject *proj,
    physis::translator::TranslationContext *tx);

//! Common subexpression elimination in grid index calculations.
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
extern void grid_index_cse(
    SgProject *proj,
    physis::translator::TranslationContext *tx);
    
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
extern void make_conditional_get_unconditional(
    SgProject *proj,
    physis::translator::TranslationContext *tx);

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis
