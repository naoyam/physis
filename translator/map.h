// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_MAP_H_
#define PHYSIS_TRANSLATOR_MAP_H_

#include "translator/translator_common.h"
#include "translator/domain.h"
#include "translator/stencil_range.h"
#include "translator/grid.h"
#include "physis/physis_util.h"

#define MAP_NAME ("PSStencilMap")

namespace physis {
namespace translator {

class TranslationContext;
//typedef map<Grid*, StencilRange> GridRangeMap;
typedef map<SgInitializedName*, StencilRange> GridRangeMap;

std::string GridRangeMapToString(GridRangeMap &gr);
// StencilRange AggregateStencilRange(GridRangeMap &gr,
//                                    const GridSet *gs);

class StencilMap {
 public:
  StencilMap(SgFunctionCallExp *call, TranslationContext *tx);

  static bool isMap(SgFunctionCallExp *call);
  static SgFunctionDeclaration *getKernelFromMapCall(SgFunctionCallExp *call);
  static SgExpression *getDomFromMapCall(SgFunctionCallExp *call);

  string toString() const;
  SgExpression *getDom() const { return dom; }
  int getID() const { return id; }
  SgFunctionDeclaration *getKernel() { return kernel; }
  int getNumDim() const { return numDim; }
  string getTypeName() const {
    return "__PSStencil" + dimStr() + "_" + kernel->get_name();
  }
  string getMapName() const {
    return "__PSStencil" + dimStr() + "Map_" + kernel->get_name();
  }
  string getRunName() const {
    return "__PSStencil" + dimStr() + "Run_" + kernel->get_name();
  }
  SgClassType*& stencil_type() { return stencil_type_; };
  SgClassDefinition *GetStencilTypeDefinition() {
    SgClassDeclaration *decl
        = isSgClassDeclaration(stencil_type()->get_declaration());
    decl = isSgClassDeclaration(decl->get_definingDeclaration());
    assert(decl);
    return decl->get_definition();
  }
  void setFunc(SgFunctionDeclaration *f) {
    assert(f);
    func = f;
  }
  SgFunctionDeclaration *getFunc() { return func; }
  SgFunctionDeclaration*& run() { return run_; }
  SgFunctionDeclaration*& run_inner() { return run_inner_; }
  SgFunctionDeclaration*& run_boundary() { return run_boundary_; }    
  const SgInitializedNamePtrList& grid_args() const { return grid_args_; }
  const SgInitializedNamePtrList& grid_params() const { return grid_params_; }  
  //GridRangeMap &gr() { return gr_; }
  GridRangeMap &grid_stencil_range_map() { return grid_stencil_range_map_; }
  const GridRangeMap &grid_stencil_range_map() const {
    return grid_stencil_range_map_; }  
  StencilRange &GetStencilRange(SgInitializedName *gv) {
    if (!isContained<SgInitializedName*, StencilRange>(
            grid_stencil_range_map(), gv)) {
      PSAssert(false);
    }
    return grid_stencil_range_map().find(gv)->second;
  }  

  // Use Kernel::isGridParamWritten and Kernel::isGridParamRead
  // bool IsWritten(Grid *g);
  // bool IsRead(Grid *g);

  SgExpressionPtrList &GetArgs() { return fc_->get_args()->get_expressions(); }

 protected:
  SgExpression *dom;
  int numDim;
  int id;
  static Counter c;
  // the kernel function
  SgFunctionDeclaration *kernel;
  // struct to hold a domain object and kernel arguments
  SgClassType *stencil_type_;
  // function to create stencil
  SgFunctionDeclaration *func;
  // function to run stencil
  SgFunctionDeclaration *run_;
  // function to run inner stencil
  SgFunctionDeclaration *run_inner_;
  // function to run boundary stencil
  SgFunctionDeclaration *run_boundary_;
  //GridRangeMap gr_;
  GridRangeMap grid_stencil_range_map_;  
  SgInitializedNamePtrList grid_args_;
  SgInitializedNamePtrList grid_params_;  
  SgFunctionCallExp *fc_;

 private:
  // NOTE: originally dimenstion is added to names, but it is probably
  // not necessry
  string dimStr() const {
    //return physis::toString(getNumDim()) + "D";
    return "";
  }
};

typedef vector<StencilMap*> StencilMapVector;

class RunKernelAttribute: public AstAttribute {
  StencilMap *stencil_map_;
  SgInitializedName *stencil_param_;
 public:
  RunKernelAttribute(StencilMap *sm,
                     SgInitializedName *stencil_param=NULL):
      stencil_map_(sm), stencil_param_(stencil_param) {}
  virtual ~RunKernelAttribute() {}
  AstAttribute *copy() {
    return new RunKernelAttribute(stencil_map_,
                                  stencil_param_);
  }
  static const std::string name;
  StencilMap *stencil_map() { return stencil_map_; };
  SgInitializedName *stencil_param() { return stencil_param_; }
};

class RunKernelLoopAttribute: public AstAttribute {
 public:
#if 0  
  RunKernelLoopAttribute(int dim, SgInitializedName *var,
                         SgExpression *begin,
                         SgExpression *end):
      dim_(dim), var_(var) {
    AddBeginExpression(begin);
    AddEndExpression(end);
  }
  RunKernelLoopAttribute(int dim, SgInitializedName *var,
                         const vector<SgExpression *> &begin,
                         const vector<SgExpression *> &end):
      dim_(dim), var_(var), begin_(begin), end_(end) {}
#endif
  RunKernelLoopAttribute(int dim, SgInitializedName *var,
                         SgExpression *begin,
                         SgExpression *end):
      dim_(dim), var_(var), begin_(begin), end_(end) {}
  
  virtual ~RunKernelLoopAttribute() {}
  AstAttribute *copy() {
    return new RunKernelLoopAttribute(dim_, var_,
                                      begin_, end_);
  }
  static const std::string name;
  int dim() const { return dim_; }
  SgInitializedName *var() { return var_; }
  SgExpression *begin() const { return begin_; }
  SgExpression *end() const { return end_; }
  SgExpression *&begin() { return begin_; }
  SgExpression *&end() { return end_; }
#if 0  
  void AddBeginExpression(SgExpression *e) {
    begin_.push_back(e);
  }
  void AddEndExpression(SgExpression *e) {
    end_.push_back(e);
  }
  void SetBeginExpression(SgExpression *e) {
    begin_.clear();
    AddBeginExpression(e);
  }
  void SetEndExpression(SgExpression *e) {
    end_.clear();
    AddEndExpression(e);
  }

  const vector<SgExpression *> &GetBeginExpressions() const {
    return begin_;
  }
  const vector<SgExpression *> &GetEndExpressions() const {
    return end_;
  }
  SgExpression *GetFirstBeginExpression() const {
    if (begin_.size() > 0) {
      return *(begin_.begin());
    } else {
      return NULL;
    }
  }
  SgExpression *GetFirstEndExpression() const {
    if (end_.size() > 0) {
      return *(end_.end());
    } else {
      return NULL;
    }
  }
#endif      
 protected:
  int dim_;
  SgInitializedName *var_;
#if 0  
  vector<SgExpression *> begin_;
  vector<SgExpression *> end_;
#endif
  SgExpression *begin_;
  SgExpression *end_;
  
};

class RunKernelLoopVarAttribute: public AstAttribute {
 public:
  RunKernelLoopVarAttribute(int dim):
      dim_(dim) {}
  virtual ~RunKernelLoopVarAttribute() {}
  AstAttribute *copy() {
    return new RunKernelLoopVarAttribute(dim_);
  }
  static const std::string name;
  int dim() const {return dim_; }
 protected:
  int dim_;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MAP_H_ */
