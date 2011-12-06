// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_KERNEL_H_
#define PHYSIS_TRANSLATOR_KERNEL_H_

#include "translator/translator_common.h"
#include "translator/grid.h"

namespace physis {
namespace translator {

struct AccessOffsets {
  int minx, maxx, miny, maxy, minz, maxz;
};

class TranslationContext;

class Kernel {
 public:
  typedef map<SgFunctionCallExp*, Kernel*> ChildMap;
 protected:
  SgFunctionDeclaration *decl;
  GridSet rGrids;
  SgInitializedNamePtrSet rGridVars;
  GridSet wGrids;
  SgInitializedNamePtrSet wGridVars;
  Kernel *parent;
  ChildMap calls;
 public:
  Kernel(SgFunctionDeclaration *decl, TranslationContext *tx,
         Kernel *k = NULL);
  const GridSet& getInGrids() const;
  const GridSet& getOutGrids() const;
  // Returns true if grid object g may be read in this
  // kernel.
  bool isRead(Grid *g) const;
  bool isReadAny(GridSet *gs) const;
  bool isGridParamRead(SgInitializedName *v) const;
  // Returns true if grid object g may be modified in this
  // kernel.
  bool isModified(Grid *g) const;
  bool isModifiedAny(GridSet *ngs) const;  
  // Returns true if variable may be modified in this kernel.
  // TODO (naoya): intra-kernel calls are not analyzed. Parameters
  // modified in inner kernels are not correctly returned about its
  // accesses. 
  bool isGridParamModified(SgInitializedName *v) const;
  bool isGridParamWritten(SgInitializedName *v) const {
    return isGridParamModified(v);
  }

  SgFunctionDeclaration *getDecl() {
    return decl;
  }

  SgInitializedNamePtrList &getArgs() {
    return decl->get_args();
  }

  SgFunctionDefinition *getDef() {
    return decl->get_definition();
  }
  void appendChild(SgFunctionCallExp *call, Kernel *child);
  std::string GetName() const { return string(decl->get_name()); }

 protected:
  void analyzeGridWrites(TranslationContext &tx);
  void analyzeGridReads(TranslationContext &tx);  
};

class RunKernelAttribute: public AstAttribute {
 public:
  RunKernelAttribute() {}
  virtual ~RunKernelAttribute() {}
  static const std::string name;  
};


} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_KERNEL_H_ */
