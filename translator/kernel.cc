// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/kernel.h"

#include <utility>
#include <boost/foreach.hpp>

#include "translator/translation_context.h"
#include "translator/alias_analysis.h"

namespace si = SageInterface;

namespace physis {
namespace translator {

static void analyzeGridAccess(SgFunctionDeclaration *decl,
                              TranslationContext &tx,
                              const set<SgInitializedName*> &gvs,
                              GridSet &grid_set,
                              SgInitializedNamePtrSet &grid_var_set) {
  SgTypePtrList relevantTypes;
  BOOST_FOREACH(SgInitializedName *gv, gvs) {
    relevantTypes.push_back(gv->get_type());
    LOG_DEBUG() << "Relevant type: " << gv->get_type()->unparseToString() << "\n";
    LOG_DEBUG() << "Used grid: "
                << gv->unparseToString() << "\n";
    const GridSet *gs = tx.findGrid(gv);
    assert(gs);
    FOREACH(it, gs->begin(), gs->end()) {
      if (*it) {
        LOG_DEBUG() << "gobj: "
                    << (*it)->toString() << "\n";
      } else {
        LOG_DEBUG() << "gobj: NULL\n";
      }
    }
    grid_set.insert(gs->begin(), gs->end());
  }

  LOG_DEBUG() << "Analyzing grid var aliases\n";
  AliasGraph ag(decl, relevantTypes);
  LOG_DEBUG() << ag;
  if (ag.hasMultipleOriginVar()) {
    LOG_ERROR() << "Grid variables must not have multiple assignments.\n";
    abort();
  }
  if (ag.hasNullInitVar()) {
    LOG_ERROR() << "Grid variables must not have NULL initialization.\n";
    abort();
  }

  BOOST_FOREACH(SgInitializedName *gv, gvs) {
    SgInitializedName* originalVar = ag.findOriginalVar(gv);
    grid_var_set.insert(originalVar);
  }
}

void Kernel::analyzeGridWrites(TranslationContext &tx) {
  SgFunctionCallExpPtrList calls =
      tx.getGridEmitCalls(decl->get_definition());
  set<SgInitializedName*> gvs;
  BOOST_FOREACH (SgFunctionCallExp *fc, calls) {
    gvs.insert(GridType::getGridVarUsedInFuncCall(fc));
  }
  analyzeGridAccess(decl, tx, gvs, wGrids, wGridVars);
}


void Kernel::analyzeGridReads(TranslationContext &tx) {
  SgFunctionCallExpPtrList calls =
      tx.getGridGetCalls(decl->get_definition());
  set<SgInitializedName*> gvs;
  BOOST_FOREACH (SgFunctionCallExp *fc, calls) {
    gvs.insert(GridType::getGridVarUsedInFuncCall(fc));
  }
  analyzeGridAccess(decl, tx, gvs, rGrids, rGridVars);
  calls =
      tx.getGridGetPeriodicCalls(decl->get_definition());
  gvs.clear();
  BOOST_FOREACH (SgFunctionCallExp *fc, calls) {
    gvs.insert(GridType::getGridVarUsedInFuncCall(fc));
  }
  analyzeGridAccess(decl, tx, gvs, rGrids, rGridVars);
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
               Kernel *parent) : parent(parent) {
  this->decl =
      isSgFunctionDeclaration(decl->get_definingDeclaration());
  assert(this->decl);
  if (si::is_C_language() || si::is_Cxx_language()) {
    analyzeGridWrites(*tx);
    analyzeGridReads(*tx);
  } else if (si::is_Fortran_language()) {
    set<SgInitializedName*> rg, wg;
    CollectionReadWriteGrids(decl->get_definition(), rg, wg);
    analyzeGridAccess(decl, *tx, rg, rGrids, rGridVars);
    analyzeGridAccess(decl, *tx, wg, wGrids, wGridVars);    
  }  
}

const GridSet& Kernel::getInGrids() const {
  return rGrids;
}

const GridSet& Kernel::getOutGrids() const {
  return wGrids;
}

#ifdef UNUSED_CODE
bool Kernel::isModified(Grid *g) const {
  if (wGrids.count(g)) return true;
  FOREACH(it, calls.begin(), calls.end()) {
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
  if (wGrids.find(NULL) != wGrids.end()) return false;
  if (wGrids.find(g) != wGrids.end()) return false;
  FOREACH(it, calls.begin(), calls.end()) {
    Kernel *child = it->second;
    if (!child->IsGridUnmodified(g)) return false;
  }
  return true;
}

bool Kernel::isGridParamModified(SgInitializedName *v) const {
  return wGridVars.find(v) != wGridVars.end();
}

#ifdef UNUSED_CODE
bool Kernel::isRead(Grid *g) const {
  if (rGrids.find(g) != rGrids.end()) return true;
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
  if (rGrids.find(NULL) != rGrids.end()) return false;
  if (rGrids.find(g) != rGrids.end()) return false;
  FOREACH(it, calls.begin(), calls.end()) {
    Kernel *child = it->second;
    if (!child->IsGridUnread(g)) return false;
  }
  return true;
}

bool Kernel::isGridParamRead(SgInitializedName *v) const {
  return rGridVars.find(v) != rGridVars.end();  
}

void Kernel::appendChild(SgFunctionCallExp *call, Kernel *child) {
  assert(calls.insert(std::make_pair(call, child)).second);
}

const std::string RunKernelAttribute::name = "RunKernel";

const std::string KernelBody::name = "KernelBody";

} // namespace translator
} // namespace physis
