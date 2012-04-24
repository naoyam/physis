// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/kernel.h"

#include <utility>

#include "translator/translation_context.h"
#include "translator/alias_analysis.h"

namespace physis {
namespace translator {

static void analyzeGridAccess(SgFunctionDeclaration *decl,
                              TranslationContext &tx,
                              SgFunctionCallExpPtrList &calls,
                              GridSet &grid_set,
                              SgInitializedNamePtrSet &grid_var_set) {
  SgTypePtrList relevantTypes;
  FOREACH(it, calls.begin(), calls.end()) {
    SgInitializedName* gv = GridType::getGridVarUsedInFuncCall(*it);
    relevantTypes.push_back(gv->get_type());
    LOG_DEBUG() << "Modified grid found: "
                << gv->unparseToString() << "\n";
    const GridSet *gs = tx.findGrid(gv);
    assert(gs);
    FOREACH(it, gs->begin(), gs->end()) {
      if (*it) {
        LOG_DEBUG() << "gobj: "
                    << (*it)->toString() << "\n";
      } else {
        LOG_WARNING() << "Ignoring possible undefined grid: "
                      << gv->unparseToString() << "\n";
        continue;
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

  FOREACH(it, calls.begin(), calls.end()) {
    SgInitializedName* gv = GridType::getGridVarUsedInFuncCall(*it);
    SgInitializedName* originalVar = ag.findOriginalVar(gv);
    grid_var_set.insert(originalVar);
  }
}

void Kernel::analyzeGridWrites(TranslationContext &tx) {
  SgFunctionCallExpPtrList calls =
      tx.getGridEmitCalls(decl->get_definition());
  analyzeGridAccess(decl, tx, calls, wGrids, wGridVars);
}

void Kernel::analyzeGridReads(TranslationContext &tx) {
  SgFunctionCallExpPtrList calls =
      tx.getGridGetCalls(decl->get_definition());
  analyzeGridAccess(decl, tx, calls, rGrids, rGridVars);
  calls =
      tx.getGridGetPeriodicCalls(decl->get_definition());
  analyzeGridAccess(decl, tx, calls, rGrids, rGridVars);
}

Kernel::Kernel(SgFunctionDeclaration *decl, TranslationContext *tx,
               Kernel *parent) : parent(parent) {
  this->decl =
      isSgFunctionDeclaration(decl->get_definingDeclaration());
  assert(this->decl);
  analyzeGridWrites(*tx);
  analyzeGridReads(*tx);
}

const GridSet& Kernel::getInGrids() const {
  return rGrids;
}

const GridSet& Kernel::getOutGrids() const {
  return wGrids;
}

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

bool Kernel::isGridParamModified(SgInitializedName *v) const {
  return wGridVars.find(v) != wGridVars.end();
}

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

bool Kernel::isGridParamRead(SgInitializedName *v) const {
  return rGridVars.find(v) != rGridVars.end();  
}

void Kernel::appendChild(SgFunctionCallExp *call, Kernel *child) {
  assert(calls.insert(std::make_pair(call, child)).second);
}

const std::string RunKernelAttribute::name = "RunKernel";

} // namespace translator
} // namespace physis
