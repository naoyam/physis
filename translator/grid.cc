// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/grid.h"
#include "translator/physis_names.h"

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
  if (tname.find(grid1dTypeKey) == 0) {
    return 1;
  } else if (tname.find(grid2dTypeKey) == 0) {
    return 2;
  } else if (tname.find(grid3dTypeKey) == 0) {
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
  if (getNumDimFromTypeName(t) == 0) return false;
  // If an underscore is used other than in the first two characters,
  // this type is not the user grid type.
  if (t.rfind("_") != 1) return false;
  return true;
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

string GridType::GetGridFuncName(SgFunctionCallExp *call) {
  if (!GridType::isGridTypeSpecificCall(call)) {
    throw PhysisException("Not a grid function call");
  }
  SgPointerDerefExp *pdref =
      isSgPointerDerefExp(call->get_function());
  SgArrowExp *exp = isSgArrowExp(pdref->get_operand());

  SgVarRefExp *rhs = isSgVarRefExp(exp->get_rhs_operand());
  assert(rhs);
  const string name = rhs->get_symbol()->get_name().getString();
  LOG_VERBOSE() << "method name: " << name << "\n";

  return name;
}

bool ValidatePointMemberType(SgType *t) {
  if (isSgArrayType(t)) {
    return ValidatePointMemberType(isSgArrayType(t)->get_base_type());
  }
  if (isSgTypedefType(t)) {
    return ValidatePointMemberType(isSgTypedefType(t)->get_base_type());
  }
  if (!(isSgTypeFloat(t) ||
        isSgTypeDouble(t))) {
    LOG_DEBUG() << "Invalid member type: " << t->unparseToString() << "\n";
    return false;
  }
  return true;
}

bool ValidatePointType(SgClassDefinition *point_def) {
  const SgDeclarationStatementPtrList &members =
      point_def->get_members();
  if (members.size() == 0) {
    LOG_ERROR() << "No struct member found.\n";
    return false;
  }
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (!member_decl) {
      LOG_ERROR() << "Unknown member val: "
                  << member_decl->unparseToString() << "\n";
      return false;
    }
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    PSAssert(vars.size() == 1);
    SgType *member_type = vars[0]->get_type();
    if (!ValidatePointMemberType(member_type)) {
      LOG_ERROR() << "Invalid point struct member: "
                  << member_decl->unparseToString() << "\n";
      return false;
    }
  }
  // Success
  return true;
}

void GridType::FindPointType() {
  const SgClassDefinition *sdef =
      isSgClassDeclaration(struct_type_->get_declaration()
                           ->get_definingDeclaration())->get_definition();
  assert(sdef);
  const SgDeclarationStatementPtrList &members =
      sdef->get_members();
  SgType *pt = NULL;
  FOREACH(it, members.begin(), members.end()) {
    SgVariableDeclaration *m = isSgVariableDeclaration(*it);
    assert(m);
    SgInitializedName *v = m->get_variables().front();
    if (v->get_name() != "_type_indicator") continue;
    pt = v->get_type();
    LOG_DEBUG() << "Grid point type: "
                << pt->unparseToString() << "\n";
  }
  if (pt == NULL) {
    LOG_ERROR() << "Point type not found.\n";
    PSAbort(1);
  }
  point_type_ = pt;
  point_def_ = NULL;
  if (!isSgClassType(point_type_)) return;
  
  SgClassDeclaration *point_decl =
      isSgClassDeclaration(
          isSgClassType(point_type_)->get_declaration());
  point_decl = isSgClassDeclaration(
      point_decl->get_definingDeclaration());
  point_def_ = point_decl->get_definition();
  PSAssert(ValidatePointType(point_def_));
  return;
}

bool GridType::IsPrimitivePointType() const {
  PSAssert(point_type_ != NULL);
  return isSgTypeFloat(point_type_) ||
      isSgTypeDouble(point_type_) ||
      isSgTypeInt(point_type_) ||
      isSgTypeLong(point_type_);
}

bool GridType::IsUserDefinedPointType() const {
  return !IsPrimitivePointType();
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
    ss << type_name_ << "_" << funcName;
  }
  return ss.str();
}

string GridType::getRealFuncName(const string &funcName,
                                 const string &kernelName) const {
  ostringstream ss;
  ss << type_name_
     << "_" << funcName
     << "_" << kernelName;
  return ss.str();
}

SgExpression *GridType::BuildElementTypeExpr() {
  SgExpression *e = NULL;
  if (isSgTypeFloat(point_type_)) {
    e = sb::buildIntVal(PS_FLOAT);
  } else if (isSgTypeDouble(point_type_)) {
    e = sb::buildIntVal(PS_DOUBLE);
  } else {
    // Assumes user-defined type
    e = sb::buildIntVal(PS_USER);
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

SgExpression *Grid::BuildAttributeExpr() {
  if (!attribute_) return NULL;
  return si::copyExpression(attribute_);
}

void Grid::SetStencilRange(const StencilRange &sr) {
  stencil_range_.merge(sr);
  return;
}

bool Grid::IsIntrinsicCall(SgFunctionCallExp *ce) {
  if (GridType::isGridTypeSpecificCall(ce)) return true;
  string funcNames[] = {
    //"grid_dimx", "grid_dimy", "grid_dimz",
    //"grid_copyin", "grid_copyout", "grid_free",
    "PSGridDim", PS_GRID_EMIT_UTYPE_NAME
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
                  << " is an intrinsic call.\n";
      return true;
    }
  }
  return false;
}

const std::string GridOffsetAttribute::name = "PSGridOffset";
const std::string GridGetAttribute::name = "PSGridGet";

GridGetAttribute::GridGetAttribute(
    SgInitializedName *gv,
    int num_dim,
    bool in_kernel,
    bool is_periodic,
    const StencilIndexList *sil,
    SgExpression *offset,
    const string &member_name):
      gv_(gv), original_gv_(gv),
      num_dim_(num_dim), in_kernel_(in_kernel),
      is_periodic_(is_periodic),
      sil_(NULL), offset_(offset),
      member_name_(member_name) {
  if (sil) {
    sil_ = new StencilIndexList(*sil);
  }
}

GridGetAttribute::GridGetAttribute(const GridGetAttribute &x):
    gv_(x.gv_), original_gv_(x.original_gv_),
    num_dim_(x.num_dim_), in_kernel_(x.in_kernel_),
    is_periodic_(x.is_periodic_),
    sil_(NULL), offset_(x.offset_),
    member_name_(x.member_name_) {
  if (x.sil_) {
    sil_ = new StencilIndexList(*x.sil_);
  }
}

GridGetAttribute::~GridGetAttribute() {
  if (sil_) {
    delete sil_;
  }
}

GridGetAttribute *GridGetAttribute::copy() {
  GridGetAttribute *a= new GridGetAttribute(*this);
  return a;
}

void GridGetAttribute::SetStencilIndexList(
    const StencilIndexList *sil) {
  if (sil) {
    if (sil_ == NULL) {
      sil_ = new StencilIndexList();
    }
    *sil_ = *sil;
  } else {
    if (sil_) {
      delete sil_;
      sil_ = NULL;
    }
  }
}

bool GridGetAttribute::IsUserDefinedType() const {
  GridType *g = rose_util::GetASTAttribute<GridType>(gv_);
  PSAssert(g);
  return g->IsUserDefinedPointType();
}


const std::string GridEmitAttribute::name = "PSGridEmit";

GridEmitAttribute::GridEmitAttribute(SgInitializedName *gv):
    gv_(gv), is_member_access_(false) {
}

GridEmitAttribute::GridEmitAttribute(SgInitializedName *gv,
                                     const string &member_name):
    gv_(gv), is_member_access_(true), member_name_(member_name) {
}

GridEmitAttribute::GridEmitAttribute(SgInitializedName *gv,
                                     const string &member_name,
                                     const vector<string> &array_offsets):
    gv_(gv), is_member_access_(true), member_name_(member_name),
    array_offsets_(array_offsets) {
}

GridEmitAttribute::GridEmitAttribute(const GridEmitAttribute &x):
    gv_(x.gv_), is_member_access_(x.is_member_access_),
    member_name_(x.member_name_), array_offsets_(x.array_offsets_) {
}

GridEmitAttribute::~GridEmitAttribute() {}

GridEmitAttribute *GridEmitAttribute::copy() {
  return new GridEmitAttribute(*this);
}




} // namespace translator
} // namespace physis
