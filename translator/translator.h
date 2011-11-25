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
#include "translator/reduce.h"

namespace physis {
namespace translator {

class TranslationContext;
class GridType;
class StencilMap;
class Run;
class Grid;

class Translator: public rose_util::RoseASTTraversal {
 public:
  Translator(const Configuration &config);
  virtual ~Translator() {}
  virtual void SetUp(SgProject *project, TranslationContext *context);  
  //! Translate the AST given by the SetUp function.
  /*!
    Call SetUp before calling this. Translator instances can be reused
    by calling Finish and SetUp.
   */
  virtual void Translate() = 0;
  //! Clear fields set for the AST set by SetUp.
  /*!
    Call this before translating different ASTs.
   */
  virtual void Finish();  
  virtual void Optimize() {}  

 protected:
  const Configuration &config_;
  SgProject *project_;
  SgSourceFile *src_;
  SgGlobal *global_scope_;
  TranslationContext *tx_;
  // types in physis_common.h
  //SgType *uvec_type_;
  SgType *ivec_type_;
  SgType *index_type_;
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

  virtual void Visit(SgFunctionCallExp *node);
  virtual void Visit(SgFunctionDeclaration *node);
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
  //! Handler for a call to reduce grids.
  /*!
    \param rd A Reduce object.
   */
  virtual void TranslateReduceGrid(Reduce *rd) {}
  //! Handler for a call to kernel reductions.
  /*!
    \param rd A Reduce object.
   */
  virtual void TranslateReduceKernel(Reduce *rd) {}
  
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
