// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/grid.h"

namespace si = SageInterface;
namespace sb = SageBuilder;


namespace physis {
namespace translator {

static const char *grid1dTypeKey = "__PSGrid1D";
static const char *grid2dTypeKey = "__PSGrid2D";
static const char *grid3dTypeKey = "__PSGrid3D";

const string GridType::name = "GridType";
const string GridType::get_name = "get";
const string GridType::get_periodic_name = "get_periodic";
const string GridType::emit_name = "emit";
const string GridType::set_name = "set";

unsigned GridType::getNumDimFromTypeName(const string &tname) {
  if (tname.find(grid1dTypeKey) != string::npos) {
    return 1;
  } else if (tname.find(grid2dTypeKey) != string::npos) {
    return 2;
  } else if (tname.find(grid3dTypeKey) != string::npos) {
    return 3;
  } else {
    return 0;
  }
}

bool GridType::isGridType(SgType *ty) {
  SgTypedefType *tt = isSgTypedefType(ty);
  // handle typedef'ed alias type too
  if (tt) {
    ty = tt->get_base_type();
  }
  if (si::isPointerType(ty)) {
    ty = si::getElementType(ty);
  }
  if (!isSgNamedType(ty)) return false;
  const string tn = isSgNamedType(ty)->get_name().getString();
  //LOG_DEBUG() << "type name: " << tn << "\n";
  return isGridType(tn);
}

bool GridType::isGridType(const string &t) {
  return getNumDimFromTypeName(t) > 0;
}

void Grid::identifySize(SgExpressionPtrList::const_iterator size_begin,
                        SgExpressionPtrList::const_iterator size_end) {
  int num_dim = gt->getNumDim();
  if (rose_util::copyConstantFuncArgs<size_t>(
          size_begin, size_end, static_size_) == num_dim) {
    has_static_size_ = true;
  } else {
    has_static_size_ = false;
  }
}

string GridType::getTypeNameFromFuncName(const string &funcName) {
  size_t p = funcName.rfind("_");
  if (p == string::npos) {
    throw PhysisException("Function name does not follow the physis "
                          "grid function naming scheme.");
  }
  return funcName.substr(0, p);
}

SgInitializedName*
GridType::getGridVarUsedInFuncCall(SgFunctionCallExp *call) {
  SgPointerDerefExp *pdref =
      isSgPointerDerefExp(call->get_function());
  if (!pdref) return false;

  SgArrowExp *exp = isSgArrowExp(pdref->get_operand());
  if (!exp) return NULL;

  SgExpression *lhs = exp->get_lhs_operand();
  SgVarRefExp *vre = isSgVarRefExp(lhs);
  if (!vre) return NULL;

  SgInitializedName *in = vre->get_symbol()->get_declaration();
  assert(in);

  if (!isGridType(in->get_type())) return NULL;

  return in;
}


bool GridType::isGridTypeSpecificCall(SgFunctionCallExp *ce) {
  return GridType::getGridVarUsedInFuncCall(ce) != NULL;
}

void GridType::findElementType() {
  SgName emit("emit");
  const SgClassDefinition *sdef =
      isSgClassDeclaration(struct_type_->get_declaration()
                           ->get_definingDeclaration())->get_definition();
  assert(sdef);
  const SgDeclarationStatementPtrList &members =
      sdef->get_members();
  elm_type_ = NULL;
  FOREACH(it, members.begin(), members.end()) {
    SgVariableDeclaration *m = isSgVariableDeclaration(*it);
    assert(m);
    SgInitializedName *v = m->get_variables().front();
    if (v->get_name() == "emit") {
      LOG_DEBUG() << "emit found\n";
    } else {
      continue;
    }
    // OG_DEBUG() << "class: " << v->get_type()->class_name() << "\n";
    SgPointerType *t = isSgPointerType(v->get_type());
    assert(t);
    SgFunctionType *emit_func_type
        = isSgFunctionType(t->get_base_type());
    SgType *emit_param_type =
        emit_func_type->get_arguments().front();
    LOG_DEBUG() << emit_param_type->class_name() << "\n";
    elm_type_ = emit_param_type;
    break;
  }
  assert(elm_type_);
  return;
}

bool GridType::isGridCall(SgFunctionCallExp *ce) {
  if (GridType::isGridTypeSpecificCall(ce)) return true;
  string funcNames[] = {
    //"grid_dimx", "grid_dimy", "grid_dimz",
    //"grid_copyin", "grid_copyout", "grid_free",
    "PSGridDim"
  };
  SgFunctionRefExp *callee = isSgFunctionRefExp(ce->get_function());
  if (!callee) {
    // this is a indirect call, which cannot be a call to grid
    // functions other than those catched by
    // isGridTypeSpecificCall.
    return false;
  }
  string calleeName = rose_util::getFuncName(callee);
  for (unsigned i = 0; i < sizeof(funcNames) / sizeof(string); i++) {
    if (calleeName == funcNames[i]) {
      LOG_DEBUG() << ce->unparseToString()
                  << " is a grid call.\n";
      return true;
    }
  }
  return false;
}

string GridType::getRealFuncName(const string &funcName) const {
  ostringstream ss;
  // element-type-independent functions
  if (funcName == "copyin" || funcName == "copyout" ||
      funcName == "dimx" || funcName == "dimy" ||
      funcName == "dimz" || funcName == "free") {
    ss << "grid" << num_dim_
       << "d_" << funcName;
  } else {
    ss << name_ << "_" << funcName;
  }
  return ss.str();
}

string GridType::getRealFuncName(const string &funcName,
                                 const string &kernelName) const {
  ostringstream ss;
  ss << name_
     << "_" << funcName
     << "_" << kernelName;
  return ss.str();
}

SgExpression *GridType::BuildElementTypeExpr() {
  SgExpression *e = NULL;
  if (isSgTypeFloat(elm_type_)) {
    e = sb::buildIntVal(PS_FLOAT);
  } else if (isSgTypeDouble(elm_type_)) {
    e = sb::buildIntVal(PS_DOUBLE);
  } else {
    PSAbort(1);
  }
  return e;
}

string Grid::toString() const {
  ostringstream ss;
  ss << "Grid object: " << gt->toString()
      // < ", stencil range: " << sr.toString();
     << ", initialized with #" << newCall;
  if (has_static_size())
    ss << ", size: " << static_size_;
  return ss.str();
}

SgExprListExp *Grid::BuildSizeExprList() {
  SgExprListExp *exp_list = sb::buildExprListExp();
  SgExpressionPtrList &args = newCall->get_args()->get_expressions();  
  int nd = gt->getNumDim();
  for (int i = 0; i < nd; ++i) {
    exp_list->append_expression(si::copyExpression(args[i]));
  }
  return exp_list;
}

SgExpression *Grid::BuildAttributeExpr() {
  if (!attribute_) return NULL;
  return si::copyExpression(attribute_);
}

const std::string GridOffsetAttribute::name = "PSGridOffset";
const std::string GridGetAttribute::name = "PSGridGet";
const std::string GridEmitAttr::name = "PSGridEmit";

} // namespace translator
} // namespace physis
