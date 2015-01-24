// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/kernel.h"

#include <utility>
#include <boost/foreach.hpp>

#include "translator/translation_context.h"
#include "translator/alias_analysis.h"

namespace si = SageInterface;

namespace physis {
namespace translator {

static void AnalyzeGridAccess(SgFunctionDeclaration *decl,
                              TranslationContext &tx,
                              const set<SgInitializedName*> &gvs,
                              GridSet &grid_set,
                              SgInitializedNamePtrSet &grid_var_set) {
  SgTypePtrList relevantTypes;
  BOOST_FOREACH(SgInitializedName *gv, gvs) {
    relevantTypes.push_back(gv->get_type());
    LOG_DEBUG() << "Relevant type: " << gv->get_type()->unparseToString() << "\n";
    LOG_DEBUG() << "Used grid: " << gv->unparseToString() << "\n";
    const GridSet *gs = tx.findGrid(gv);
    assert(gs);
    FOREACH(it, gs->begin(), gs->end()) {
      if (*it) {
        LOG_DEBUG() << "gobj: " << (*it)->toString() << "\n";
      } else {
        LOG_DEBUG() << "gobj: NULL\n";
      }
    }
    grid_set.insert(gs->begin(), gs->end());
  }

  LOG_DEBUG() << "Analyzing grid var aliases\n";
  AliasGraph ag(decl, relevantTypes);
  LOG_DEBUG() << ag;
  if (ag.HasMultipleOriginVar()) {
    LOG_ERROR() << "Grid variables must not have multiple assignments.\n";
    abort();
  }
  if (ag.HasNullInitVar()) {
    LOG_ERROR() << "Grid variables must not have NULL initialization.\n";
    abort();
  }

  BOOST_FOREACH(SgInitializedName *gv, gvs) {
    SgInitializedName* originalVar = ag.FindOriginalVar(gv);
    grid_var_set.insert(originalVar);
  }
}

void Kernel::AnalyzeGridWrites(TranslationContext &tx) {
  SgFunctionCallExpPtrList calls =
      tx.getGridEmitCalls(decl_->get_definition());
  set<SgInitializedName*> gvs;
  BOOST_FOREACH (SgFunctionCallExp *fc, calls) {
    gvs.insert(GridType::getGridVarUsedInFuncCall(fc));
  }
  AnalyzeGridAccess(decl_, tx, gvs, wGrids_, wGridVars_);
}

void Kernel::AnalyzeGridReads(TranslationContext &tx) {
  SgFunctionCallExpPtrList calls =
      tx.getGridGetCalls(decl_->get_definition());
  set<SgInitializedName*> gvs;
  BOOST_FOREACH (SgFunctionCallExp *fc, calls) {
    gvs.insert(GridType::getGridVarUsedInFuncCall(fc));
  }
  AnalyzeGridAccess(decl_, tx, gvs, rGrids_, rGridVars_);
  calls =
      tx.getGridGetPeriodicCalls(decl_->get_definition());
  gvs.clear();
  BOOST_FOREACH (SgFunctionCallExp *fc, calls) {
    gvs.insert(GridType::getGridVarUsedInFuncCall(fc));
  }
  AnalyzeGridAccess(decl_, tx, gvs, rGrids_, rGridVars_);
}

static void CollectionReadWriteGrids(SgFunctionDefinition *fdef,
                                     set<SgInitializedName*> &read_grids,
                                     set<SgInitializedName*> &written_grids) {
  vector<SgDotExp*> grid_reads =
      si::querySubTree<SgDotExp>(fdef);
  BOOST_FOREACH(SgDotExp *de, grid_reads) {
    SgVarRefExp *gr = isSgVarRefExp(de->get_lhs_operand());
    PSAssert(gr);
    SgInitializedName *gv = si::convertRefToInitializedName(gr);
    if (!rose_util::GetASTAttribute<GridType>(gv)) {
      // Not a grid 
      continue;
    }
    LOG_DEBUG() << "Access to grid var found: "
                << de->unparseToString() << "\n";
    // write access if located as lhs of assign op
    if (isSgAssignOp(de->get_parent()) &&
        isSgAssignOp(de->get_parent())->get_lhs_operand() == de) {
      written_grids.insert(gv);
    } else {
      read_grids.insert(gv);
    }
  }
}

Kernel::Kernel(SgFunctionDeclaration *decl, TranslationContext *tx,
               Kernel *parent) : parent_(parent) {
  this->decl_ =
      isSgFunctionDeclaration(decl->get_definingDeclaration());
  assert(this->decl_);
  if (si::is_C_language() || si::is_Cxx_language()) {
    AnalyzeGridWrites(*tx);
    AnalyzeGridReads(*tx);
  } else if (si::is_Fortran_language()) {
    set<SgInitializedName*> rg, wg;
    CollectionReadWriteGrids(decl->get_definition(), rg, wg);
    AnalyzeGridAccess(decl, *tx, rg, rGrids_, rGridVars_);
    AnalyzeGridAccess(decl, *tx, wg, wGrids_, wGridVars_);    
  }  
}

Kernel::Kernel(const Kernel &k):
    decl_(k.decl_), rGrids_(k.rGrids_), rGridVars_(k.rGridVars_),
    wGrids_(k.wGrids_), wGridVars_(k.wGridVars_), parent_(k.parent_),
    calls_(k.calls_) {
}

const GridSet& Kernel::GetInGrids() const {
  return rGrids_;
}

const GridSet& Kernel::GetOutGrids() const {
  return wGrids_;
}

#ifdef UNUSED_CODE
bool Kernel::isModified(Grid *g) const {
  if (wGrids_.count(g)) return true;
  FOREACH(it, calls_.begin(), calls_.end()) {
    Kernel *child = it->second;
    if (child->isModified(g)) return true;
  }
  return false;
}

bool Kernel::isModifiedAny(GridSet *gs) const {
  FOREACH (it, gs->begin(), gs->end()) {
    if (isModified(*it)) return true;
  }
  return false;
}
#endif

bool Kernel::IsGridUnmodified(Grid *g) const {
  assert(g != NULL);
  // If NULL is contained, g may be modified.
  if (wGrids_.find(NULL) != wGrids_.end()) return false;
  if (wGrids_.find(g) != wGrids_.end()) return false;
  FOREACH(it, calls_.begin(), calls_.end()) {
    Kernel *child = it->second;
    if (!child->IsGridUnmodified(g)) return false;
  }
  return true;
}

bool Kernel::IsGridParamModified(SgInitializedName *v) const {
  return wGridVars_.find(v) != wGridVars_.end();
}

#ifdef UNUSED_CODE
bool Kernel::isRead(Grid *g) const {
  if (rGrids_.find(g) != rGrids_.end()) return true;
  FOREACH(it, calls.begin(), calls.end()) {
    Kernel *child = it->second;
    if (child->isRead(g)) return true;
  }
  return false;
}

bool Kernel::isReadAny(GridSet *gs) const {
  FOREACH (it, gs->begin(), gs->end()) {
    if (isRead(*it)) return true;
  }
  return false;
}
#endif

bool Kernel::IsGridUnread(Grid *g) const {
  if (rGrids_.find(NULL) != rGrids_.end()) return false;
  if (rGrids_.find(g) != rGrids_.end()) return false;
  FOREACH(it, calls_.begin(), calls_.end()) {
    Kernel *child = it->second;
    if (!child->IsGridUnread(g)) return false;
  }
  return true;
}

bool Kernel::IsGridParamRead(SgInitializedName *v) const {
  return rGridVars_.find(v) != rGridVars_.end();  
}

void Kernel::AppendChild(SgFunctionCallExp *call, Kernel *child) {
  assert(calls_.insert(std::make_pair(call, child)).second);
}

const std::string RunKernelAttribute::name = "RunKernel";

const std::string Kernel::name = "Kernel";

const std::string KernelBody::name = "KernelBody";

} // namespace translator
} // namespace physis
