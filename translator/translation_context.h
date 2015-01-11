// Licensed under the BSD license. See LICENSE.txt for more details.

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
  typedef map<SgExpression*, StencilMap*> MapCallMap;
  typedef map<SgFunctionCallExp*, Run*> RunMap;

 public:
  explicit TranslationContext(SgProject *project): project_(project) {
    Build();
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

  KernelMap entry_kernels_;
  KernelMap inner_kernels_;

  SgType *dom1d_type_;
  SgType *dom2d_type_;
  SgType *dom3d_type_;
  //SgType *f_dom_type_;

  MapCallMap stencil_map_;

  RunMap run_map_;

  void Build();
  // No dependency
  void AnalyzeGridTypes();
  // Depends on analyzeDomainExpr, analyzeMap
  void AnalyzeGridVars();
  // No dependency
  void AnalyzeDomainExpr();
  // No dependency
  void AnalyzeMap();
  void locateDomainTypes();
  // Depends on analyzeMap, analyzeGridVars
  void AnalyzeKernelFunctions();
  void AnalyzeRun();
  // Depends on analyzeGridVars, analyzeKernelFunctions
  //void MarkReadWriteGrids();

  //! Find and collect information on reductions
  void AnalyzeReduce();

 public:

  // Accessors
  SgProject *project() { return project_; }
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
  
  void registerMap(SgExpression *e, StencilMap *mc) {
    assert(stencil_map_.insert(std::make_pair(e, mc)).second);
  }

  bool IsMap(SgExpression *e) {
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
  SgVarRefExp *IsFree(SgFunctionCallExp *ce);
  SgVarRefExp *IsCopyin(SgFunctionCallExp *ce);
  SgVarRefExp *IsCopyout(SgFunctionCallExp *ce);

  SgFunctionCallExpPtrList getGridEmitCalls(SgScopeStatement *scope);
  SgFunctionCallExpPtrList getGridGetCalls(SgScopeStatement *scope);
  SgFunctionCallExpPtrList getGridGetPeriodicCalls(SgScopeStatement *scope);  

  void print(std::ostream &os) const;

  CallGridMap &grid_new_map() { return grid_new_map_; }

  bool IsInit(SgFunctionCallExp *call) const;

 protected:
  SgFunctionCallExpPtrList getGridCalls(SgScopeStatement *scope,
                                        string methodName);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_TRANSLATION_CONTEXT_H_ */
