// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_GRID_H_
#define PHYSIS_TRANSLATOR_GRID_H_

#include <set>

#include "translator/translator_common.h"
#include "physis/physis_util.h"
#include "translator/rose_util.h"
#include "translator/stencil_range.h"

namespace physis {
namespace translator {

static const char *gridIndexNames[3] = {"x", "y", "z"};

// Represents a grid type, not a particular grid object.
// Grid objects are handled by class Grid.
class GridType: public AstAttribute {
  SgClassType *struct_type_;
  SgTypedefType *user_type_;
  unsigned num_dim_;
  string name_;
  SgType *elm_type_;

 public:
  GridType(SgClassType *struct_type, SgTypedefType *user_type)
      : struct_type_(struct_type), user_type_(user_type) {
    name_ = user_type_->get_name().getString();
    LOG_DEBUG() << "grid type name: " << name_ << "\n";
    string realName = struct_type_->get_name().getString();
    num_dim_ = getNumDimFromTypeName(realName);
    LOG_DEBUG() << "grid dimension: " << num_dim_ << "\n";
    findElementType();
    LOG_DEBUG() << "grid element type: "
                << elm_type_->class_name() << "\n";
  }

  unsigned getNumDim() const {
    return num_dim_;
  }
  unsigned num_dim() const { return num_dim_; }
  const string& getName() const {
    return name_;
  }
  SgType *getElmType() const {
    return elm_type_;
  }
  SgType *elm_type() const { return elm_type_; }
  
  string getRealFuncName(const string &funcName) const;
  
  string getRealFuncName(const string &funcName,
                         const string &kernelName) const;
  string toString() const {
    return name_;
  }

  string getNewName() const {
    return name_ + "New";
  }

  static string getTypeNameFromFuncName(const string &funcName);
  static unsigned getNumDimFromTypeName(const string &tname);
  static bool isGridType(const SgType *ty);
  static bool isGridType(const string &t);
  static bool isGridTypeSpecificCall(SgFunctionCallExp *ce);
  static SgInitializedName*
  getGridVarUsedInFuncCall(SgFunctionCallExp *call);
  static bool isGridCall(SgFunctionCallExp *ce);
  SgExpression *BuildElementTypeExpr();
  static const string name;
  static const string get_name;
  static const string get_periodic_name;
  static const string emit_name;
  static const string set_name;  
 private:
  void findElementType();
};

class Grid {
  GridType *gt;
  SgFunctionCallExp *newCall;
  //StencilRange sr;
  //vector<unsigned int> sizes;
  IntVector static_size_;
  bool has_static_size_;
  void identifySize(SgExpressionPtrList::const_iterator size_begin,
                    SgExpressionPtrList::const_iterator size_end);
  bool _isReadWrite;
  SgExpression *attribute_;
  
 public:
  
  Grid(GridType *gt, SgFunctionCallExp *newCall):
      gt(gt), newCall(newCall), 
      _isReadWrite(false), attribute_(NULL) {
    SgExpressionPtrList &args = newCall->get_args()->get_expressions();
    size_t num_dims = gt->getNumDim();
    PSAssert(args.size() == num_dims ||
             args.size() == num_dims+1);
    if (num_dims+1 == args.size()) {
      // grid attribute is given
      attribute_ = args[num_dims];
      LOG_DEBUG() << "Attribute is specified: "
                  << attribute_->unparseToString() << "\n";
    }
    
    identifySize(args.begin(), args.begin() + num_dims);
    if (has_static_size())
      LOG_DEBUG() << "static grid generated: "
                  << toString() << "\n";
  }

  GridType *getType() {
    return gt;
  }
  virtual ~Grid() {}
  string toString() const;
  int getNumDim() const {
    return gt->getNumDim();
  }
  bool has_static_size() const {
    return has_static_size_;
  }
  const IntVector &static_size() const {
    assert(has_static_size());
    return static_size_;
  }

  template <class I>
  string getStaticGlobalOffset(I offsets) const {
    // For some reason, index expression with z offset
    // appearing first results in faster CUDA code
    vector<unsigned int>::const_iterator sizes = static_size().begin();
    StringJoin sj("+");
    StringJoin sizeStr("*");
    list<string> t;
    for (unsigned i = 0; i < getNumDim(); ++i, ++offsets) {
      string goffset = "(" + string(gridIndexNames[i]) +
          "+(" + *offsets + "))";
      if (i == 0) {
        t.push_back(goffset);
      } else {
        t.push_back(goffset + "*" + sizeStr.get());
      }
      sizeStr << *sizes;
      ++sizes;
    }
    FOREACH(it, t.rbegin(), t.rend()) {
      sj << *it;
    }
    return sj.get();
  }

  bool isReadWrite() const {
    return _isReadWrite;
  }
  void setReadWrite(bool b = true) {
    _isReadWrite = b;
  }
  SgExprListExp *BuildSizeExprList();
  SgExpression *BuildAttributeExpr();
};

typedef std::set<Grid*> GridSet;

class GridOffsetAttribute: public AstAttribute {
 public:
  GridOffsetAttribute(int num_dim, bool periodic,
                      const vector<SgExpression *> &indices):
      num_dim_(num_dim), periodic_(periodic), indices_(indices) {
  }
  GridOffsetAttribute(int num_dim, bool periodic):
      num_dim_(num_dim), periodic_(periodic) {
  }
  virtual ~GridOffsetAttribute() {}
  AstAttribute *copy() {
    GridOffsetAttribute *a= new GridOffsetAttribute(num_dim_, periodic_,
                                                    indices_);
    return a;
  }
  void AppendIndex(SgExpression *index) {
    indices_.push_back(index);
  }
  static const std::string name;
  SgExpression *GetIndexAt(int dim) { return indices_.at(dim-1); }
  SgExpressionVector &indices() { return indices_; }
  bool periodic() const { return periodic_; }
  
 protected:
  int num_dim_;
  bool periodic_;  
  SgExpressionVector indices_;
};


class GridGetAttribute: public AstAttribute {
 public:
  GridGetAttribute(SgInitializedName *gv,
                   int num_dim,
                   bool in_kernel,
                   const StencilIndexList &sil,
                   SgExpression *offset=NULL):
      gv_(gv), num_dim_(num_dim), in_kernel_(in_kernel), sil_(sil),
      offset_(offset) {
  }
  virtual ~GridGetAttribute() {}
  AstAttribute *copy() {
    GridGetAttribute *a= new GridGetAttribute(gv_, num_dim_,
                                              in_kernel_, sil_,
                                              offset_);
    return a;
  }
  static const std::string name;
  bool in_kernel() const { return in_kernel_; }
  void SetInKernel(bool t) { in_kernel_ = t; };
  SgInitializedName *gv() const { return gv_; }
  SgInitializedName *&gv() { return gv_; }
  void SetStencilIndexList(const StencilIndexList &sil) {
    sil_ = sil;
  }
  const StencilIndexList &GetStencilIndexList() { return sil_; }
  int num_dim() const { return num_dim_; }
  void SetOffset(SgExpression *offset) { offset_ = offset; }
  SgExpression *offset() const { return offset_; }
  SgExpression *&offset() { return offset_; }
  
 protected:
  SgInitializedName *gv_;
  int num_dim_;  
  bool in_kernel_;
  StencilIndexList sil_;
  SgExpression *offset_;

};

class GridEmitAttr: public AstAttribute {
 public:
  static const std::string name;
};

} // namespace translator
} // namespace physis

inline std::ostream &operator<<(std::ostream &os, 
                                const physis::translator::Grid &g) {
  return os << g.toString();
}


#endif /* PHYSIS_TRANSLATOR_GRID_H_ */
