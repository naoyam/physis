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
/*
 * @proj The whole AST.
 * @param tx The translation context built for the AST.
 */
extern void null_optimization(
    SgProject *proj,
    physis::translator::TranslationContext *tx);


} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis
