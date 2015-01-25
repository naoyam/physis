// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MAP_H_
#define PHYSIS_TRANSLATOR_MAP_H_

#include "translator/translator_common.h"
#include "translator/domain.h"
#include "translator/stencil_range.h"
#include "translator/grid.h"
#include "physis/physis_util.h"

namespace physis {
namespace translator {

using std::pair;

class TranslationContext;

typedef pair<SgInitializedName*, string> GridMember;
typedef map<GridMember, StencilRange> GridMemberRangeMap;

class StencilMap {
 public:
  enum Type {kNormal, kRedBlack, kRed, kBlack};
  
  StencilMap(SgFunctionCallExp *call, TranslationContext *tx);

  static bool IsMap(SgFunctionCallExp *call);
  static Type AnalyzeType(SgFunctionCallExp *call);
  static SgFunctionDeclaration *getKernelFromMapCall(SgFunctionCallExp *call);
  static SgExpression *getDomFromMapCall(SgFunctionCallExp *call);

  string toString() const;
  SgExpression *getDom() const { return dom; }
  int getID() const { return id; }
  SgFunctionDeclaration *getKernel() { return kernel; }
  int getNumDim() const { return numDim; }

  string GetTypeName() const;
  string GetMapName() const;
  string GetRunName() const;

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

  // Use Kernel::isGridParamWritten and Kernel::isGridParamRead
  // bool IsWritten(Grid *g);
  // bool IsRead(Grid *g);

  SgExpressionPtrList &GetArgs() { return fc_->get_args()->get_expressions(); }

  //! Returns true if a grid is accessed with get_periodic.
  /*!
    \param gv Grid param name.
    \return True if the grid is accessed with get_periodic.
   */
  bool IsGridPeriodic(SgInitializedName *gv) const;
  //! Marks a grid as accessed with get_periodic.
  /*!
    \param gv Grid param name.
  */
  void SetGridPeriodic(SgInitializedName *gv);

  //! Returns true if red-black stencil is used.
  bool IsRedBlack() const {
    return type_ == kRedBlack;
  }

  bool IsRed() const {
    return type_ == kRed;
  }

  bool IsBlack() const {
    return type_ == kBlack;
  }

  bool IsRedBlackVariant() const {
    return IsRedBlack() || IsRed() || IsBlack();
  }

 protected:
  Type type_;
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
  SgInitializedNamePtrList grid_args_;
  SgInitializedNamePtrList grid_params_;  
  SgFunctionCallExp *fc_;
  std::set<SgInitializedName*> grid_periodic_set_;
  

 private:
  // NOTE: originally dimenstion is added to names, but it is probably
  // not necessry
  string dimStr() const {
    //return physis::toString(getNumDim()) + "D";
    return "";
  }
  static string GetInternalNamePrefix();

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
  RunKernelAttribute *copy() {
    return new RunKernelAttribute(stencil_map_, stencil_param_);
  }
  static const std::string name;
  StencilMap *stencil_map() { return stencil_map_; };
  SgInitializedName *stencil_param() { return stencil_param_; }
};

class RunKernelLoopAttribute: public AstAttribute {
 public:
  enum Kind {
    MAIN, FIRST, LAST
  };
  RunKernelLoopAttribute(int dim, Kind kind=MAIN):
      dim_(dim), kind_(kind)  {}
  virtual ~RunKernelLoopAttribute() {}
  RunKernelLoopAttribute *copy() {
    return new RunKernelLoopAttribute(dim_, kind_);
  }
  static const std::string name;
  int dim() const { return dim_; }
  void SetMain() { kind_ = MAIN; }
  void SetFirst() { kind_ = FIRST; }
  void SetLast() { kind_ = LAST; }
  bool IsMain() { return kind_ == MAIN; }
  bool IsFrist() { return kind_ == FIRST; }
  bool IsLast() { return kind_ == LAST; }    
 protected:
  int dim_;
  Kind kind_;
};

class KernelLoopAnalysis {
 public:
  static SgVarRefExp *GetLoopVar(SgForStatement *loop);
  static SgExpression *GetLoopBegin(SgForStatement *loop);
  static SgExpression *GetLoopEnd(SgForStatement *loop);  
};

class RunKernelIndexVarAttribute: public AstAttribute {
 public:
  RunKernelIndexVarAttribute(int dim):
      dim_(dim) {}
  virtual ~RunKernelIndexVarAttribute() {}
  RunKernelIndexVarAttribute *copy() {
    return new RunKernelIndexVarAttribute(dim_);
  }
  static const std::string name;
  int dim() const {return dim_; }
 protected:
  int dim_;
};

/** attribute for kernel caller */
class RunKernelCallerAttribute: public AstAttribute {
 public:
  RunKernelCallerAttribute() {}
  virtual ~RunKernelCallerAttribute() {}
  RunKernelCallerAttribute *copy() {
    return new RunKernelCallerAttribute();
  }
  static const std::string name;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MAP_H_ */
