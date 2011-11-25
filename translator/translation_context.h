// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_TRANSLATION_CONTEXT_H_
#define PHYSIS_TRANSLATOR_TRANSLATION_CONTEXT_H_

#include <map>
#include <ostream>
#include <set>
#include <utility>

#include "translator/translator_common.h"
#include "translator/grid.h"
#include "translator/domain.h"
#include "translator/map.h"
#include "translator/run.h"
#include "translator/reduce.h"
// Why this is included? 
//#include "translator/CallGraph.h"
#include "translator/kernel.h"

using std::map;
using std::set;

namespace physis {
namespace translator {

class TranslationContext {
 public:
  typedef map<SgType*, GridType*> GridTypeMap;
  typedef map<SgInitializedName*, GridSet> GridVarMap;
  typedef map<SgFunctionCallExp*, Grid*> CallGridMap;
  typedef map<SgFunctionDeclaration*, Kernel*> KernelMap;
  typedef map<SgExpression*, DomainSet> DomainMap;
  typedef map<SgExpression*, StencilMap*> MapCallMap;
  typedef map<SgFunctionCallExp*, Run*> RunMap;
  typedef map<SgFunctionCallExp*, StencilIndexList> StencilIndexMap;

 public:
  explicit TranslationContext(SgProject *project): project_(project) {
    build();
  }

 protected:
  SgProject *project_;
  SgIncidenceDirectedGraph *call_graph_;
  // type-grid mapping. grid objects only contain information
  // generic to the particular type without known sizes
  GridTypeMap grid_type_map_;
  // variable-grid mapping. use this to find a variable-specific
  // grid object, which can have static constant size
  GridVarMap grid_var_map_;
  // grid_new call -> grid object
  CallGridMap grid_new_map_;
  // AST expression -> domain object
  // Note: grid objects are associated with variables rather
  // than expressions. Domains used as arguments to function
  // calls can be an call expression, so this mapping needs to
  // be able to hold mapping from expressions. GridMap can also
  // be changed to hold mapping from expressions to grid
  // objects.
  DomainMap domain_map_;
  // grid calls (e.g., grid_get) -> grid object
  // allGridMap gridCallMap;

  KernelMap entry_kernels_;
  KernelMap inner_kernels_;

  SgType *dom1d_type_;
  SgType *dom2d_type_;
  SgType *dom3d_type_;

  MapCallMap stencil_map_;

  RunMap run_map_;

  // Mapping from grid_get to its StencilIndex
  StencilIndexMap stencil_indices_;

  void build();
  // No dependency
  void analyzeGridTypes();
  // Depends on analyzeDomainExpr, analyzeMap
  void analyzeGridVars(DefUseAnalysis &dua);
  // No dependency
  void analyzeDomainExpr(DefUseAnalysis &dua);
  // No dependency
  void analyzeMap();
  void locateDomainTypes();
  // Depends on analyzeMap, analyzeGridVars
  void analyzeKernelFunctions();
  void analyzeRun(DefUseAnalysis &dua);
  // Depends on analyzeGridVars, analyzeKernelFunctions
  void markReadWriteGrids();

  //! Find and collect information on reductions
  void AnalyzeReduce();

 public:

  // Accessors
  RunMap &run_map() { return run_map_; }

  
  void registerGridType(SgType *t, GridType *g) {
    grid_type_map_.insert(std::make_pair(t, g));
  }

  GridType *findGridType(SgType *t) {
    GridTypeMap::iterator it = grid_type_map_.find(t);
    if (it == grid_type_map_.end()) {
      return NULL;
    } else {
      return it->second;
    }
  }
  GridType *findGridType(SgInitializedName *in) {
    return findGridType(in->get_type());
  }
  GridType *findGridType(SgVarRefExp *gv) {
    SgInitializedName *gin = gv->get_symbol()->get_declaration();
    return findGridType(gin);
  }

  GridTypeMap::iterator gridTypeBegin() {
    return grid_type_map_.begin();
  }

  GridTypeMap::iterator gridTypeEnd() {
    return grid_type_map_.end();
  }

  GridType *findGridTypeByNew(const string &fname);

  template <class T, class S>
  bool makeAssociation(T &src, S &dst, map<T, set<S> > &m) {
    set<S> &s = m[src];
    return s.insert(dst).second;
  }

  bool associateVarWithGrid(SgInitializedName *var, Grid *g) {
    // g can be a NULl when var is declared without
    // initialization
    assert(var);
    return makeAssociation(var, g, grid_var_map_);
  }

  const GridSet *findGrid(SgInitializedName *var) const;
  // newCall: the call to create this grid object
  Grid *getOrCreateGrid(SgFunctionCallExp *newCall);

  Grid *findGrid(SgFunctionCallExp *newCall);
  bool registerInnerKernel(SgFunctionDeclaration *fd,
                           SgFunctionCallExp *call,
                           Kernel *parentKernel);
  bool registerEntryKernel(SgFunctionDeclaration *fd);
  bool isEntryKernel(SgFunctionDeclaration *fd);
  bool isInnerKernel(SgFunctionDeclaration *fd);
  bool isKernel(SgFunctionDeclaration *fd);
  Kernel *findKernel(SgFunctionDeclaration *fd);
  bool associateExpWithDomain(SgExpression *exp, Domain *d);
  const DomainSet *findDomainAll(SgExpression *exp);
  Domain *findDomain(SgExpression *exp);
  DomainMap &domain_map() { return domain_map_; }

  void registerMap(SgExpression *e, StencilMap *mc) {
    assert(stencil_map_.insert(std::make_pair(e, mc)).second);
  }

  bool isMap(SgExpression *e) {
    return stencil_map_.find(e) != stencil_map_.end();
  }

  StencilMap *findMap(SgExpression *e);

  MapCallMap::iterator mapBegin() {
    return stencil_map_.begin();
  }

  MapCallMap::iterator mapEnd() {
    return stencil_map_.end();
  }

  void registerRun(SgFunctionCallExp *call, Run *x) {
    assert(run_map_.insert(std::make_pair(call, x)).second);
  }

  bool isRun(SgFunctionCallExp *call) {
    return run_map_.find(call) != run_map_.end();
  }

  Run *findRun(SgFunctionCallExp *call) {
    RunMap::iterator it = run_map_.find(call);
    if (it == run_map_.end()) {
      return NULL;
    } else {
      return it->second;
    }
  }


  bool isNewCall(SgFunctionCallExp *ce);
  bool isNewFunc(const string &funcName);


  //! Returns a Reduce attribute object for a reduce call.
  /*!
    \param call A function call.
    \return The Reduce object for the call when it is a call to Reduce
    intrinsic. NULL otherwise.
   */
  Reduce *GetReduce(SgFunctionCallExp *call) const;

  // ool isGridTypeSpecificCall(SgFunctionCallExp *ce);
  // gInitializedName* getGridVarUsedInFuncCall(SgFunctionCallExp *call);
  string getGridFuncName(SgFunctionCallExp *call);

  SgFunctionCallExpPtrList getGridEmitCalls(SgScopeStatement *scope);
  SgFunctionCallExpPtrList getGridGetCalls(SgScopeStatement *scope);

  void print(std::ostream &os) const;

  CallGridMap &grid_new_map() { return grid_new_map_; }

  bool IsInit(SgFunctionCallExp *call) const;

  void registerStencilIndex(SgFunctionCallExp *call, const StencilIndexList &sil);
  const StencilIndexList* findStencilIndex(SgFunctionCallExp *call);

 protected:
  SgFunctionCallExpPtrList getGridCalls(SgScopeStatement *scope,
                                        string methodName);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_TRANSLATION_CONTEXT_H_ */
