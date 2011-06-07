// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_TRANSLATOR_H_

#include "translator/translator_common.h"
#include "translator/rose_traversal.h"
#include "translator/configuration.h"

namespace physis {
namespace translator {

class TranslationContext;
class GridType;
class StencilMap;
class Run;
class Grid;

class Translator: public RoseASTTraversal {
 public:
  Translator(const Configuration &config):
      config_(config), grid_type_name_("__PSGrid") {}
  virtual ~Translator() {}
  // This is the public interface to run the the
  // translator. Method run with no parameters is the actual
  // implementation.
  void run(SgProject *project, TranslationContext *context);
 protected:
  const Configuration &config_;
  SgProject *project_;
  SgSourceFile *src_;
  SgGlobal *global_scope_;
  TranslationContext *tx_;
  // types in physis_common.h
  //SgType *uvec_type_;
  SgType *ivec_type_;
  SgClassDeclaration *grid_decl_;
  SgTypedefType *grid_type_;
  SgType *grid_ptr_type_;
  SgTypedefType *dom_type_;
  SgType *dom_ptr_type_;
  //SgFunctionDeclaration *grid_swap_decl_;
  SgFunctionSymbol *grid_swap_;
  SgFunctionSymbol *grid_dim_get_func_;

  string grid_type_name_;
  string target_specific_macro_;

  virtual void buildGridDecl();

  // Concrete classes implement this method
  virtual void run() = 0;
  virtual void optimize() {}
  virtual void finish() {}
  virtual void visit(SgClassDeclaration *node) {}
  virtual void visit(SgFunctionCallExp *node);
  virtual void visit(SgFunctionDeclaration *node);
  virtual void translateKernelDeclaration(SgFunctionDeclaration *node) {}
  virtual void translateInit(SgFunctionCallExp *node) {}
  virtual void translateNew(SgFunctionCallExp *node,
                            GridType *gt) {}
  virtual void translateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool isKernel) {}
  // Returns true if translation is done. If this function returns
  // false, translateGet is used.
  virtual bool translateGetHost(SgFunctionCallExp *node,
                                SgInitializedName *gv) {
    return false;
  }
  // Returns true if translation is done. If this function returns
  // false, translateGet is used.  
  virtual bool translateGetKernel(SgFunctionCallExp *node,
                                  SgInitializedName *gv) {
    return false;
  } 
  virtual void translateEmit(SgFunctionCallExp *node,
                             SgInitializedName *gv) {}
  virtual void translateSet(SgFunctionCallExp *node,
                            SgInitializedName *gv) {}
  virtual void translateGridCall(SgFunctionCallExp *node,
                                 SgInitializedName *gv) {}
  virtual void translateMap(SgFunctionCallExp *node,
                            StencilMap *s) {}
  virtual void translateRun(SgFunctionCallExp *node,
                            Run *run) {}
  void defineMacro(const string &name, const string &val="");

  SgClassDeclaration *getDomainDeclaration() {
    SgClassType *t = isSgClassType(dom_type_->get_base_type());
    SgClassDeclaration *d =
        isSgClassDeclaration(t->get_declaration());
    assert(d);
    return d;
  }
};

} // namespace translator
} // namespace physis

#endif /* TRANSLATOR_H_ */
