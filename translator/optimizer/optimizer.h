// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_OPTIMIZER_OPTIMIZER_H_
#define PHYSIS_TRANSLATOR_OPTIMIZER_OPTIMIZER_H_

#include "translator/translator_common.h"
#include "translator/translation_context.h"
#include "translator/configuration.h"
#include "translator/builder_interface.h"

namespace physis {
namespace translator {
namespace optimizer {

class Optimizer {
 public:
  Optimizer(SgProject *proj,
            physis::translator::TranslationContext *tx,
            physis::translator::BuilderInterface *builder,
            physis::translator::Configuration *config)
      : proj_(proj), tx_(tx), builder_(builder), config_(config) {
    if (config_->LookupFlag("OPT_OFFSET_COMP")) {
      config_->SetFlag("OPT_OFFSET_CSE", true);
      config_->SetFlag("OPT_OFFSET_SPATIAL_CSE", true);
    }
    if (config_->LookupFlag("OPT_LOOP_PEELING") ||
        config_->LookupFlag("OPT_REGISTER_BLOCKING") ||
        config_->LookupFlag("OPT_OFFSET_CSE") ||
        config_->LookupFlag("OPT_OFFSET_SPATIAL_CSE") ||        
        config_->LookupFlag("OPT_LOOP_OPT")) {
      config_->SetFlag("OPT_KERNEL_INLINING", true);
    }
    if (config_->LookupFlag("OPT_REGISTER_BLOCKING")) {
      config_->SetFlag("OPT_LOOP_PEELING", true);
    }
    if (config_->LookupFlag("OPT_LOOP_OPT")) {
      config_->SetFlag("OPT_OFFSET_CSE", true);
      config_->SetFlag("OPT_OFFSET_SPATIAL_CSE", true);
    }
  }
  virtual ~Optimizer() {}
  //! Pre-translation optimizations
  virtual void Stage1();
  //! Post-translation optimizations
  virtual void Stage2();
 protected:
  SgProject *proj_;  
  physis::translator::TranslationContext *tx_;
  physis::translator::BuilderInterface *builder_;
  physis::translator::Configuration *config_;
  virtual void DoStage1();
  virtual void DoStage2();  
  virtual void Stage1PreProcess();
  virtual void Stage1PostProcess();
  virtual void Stage2PreProcess();
  virtual void Stage2PostProcess();
};

} // namespace optimizer
} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_OPTIMIZER_OPTIMIZER_H_ */
