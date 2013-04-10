// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/cuda_runtime_builder.h"

#include "translator/translation_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

SgExpression *CUDARuntimeBuilder::BuildGridRefInRunKernel(
    SgInitializedName *gv,
    SgFunctionDeclaration *run_kernel) {
  // Find the parameter with the same name as gv.
  // TODO:
  const SgInitializedNamePtrList &plist =
      run_kernel->get_parameterList()->get_args();
  FOREACH (plist_it, plist.begin(), plist.end()) {
    SgInitializedName *pin = *plist_it;
    if (pin->get_name() == gv->get_name()) {
      // The grid parameter with the same name as gv found
      SgVariableSymbol *ps =
          isSgVariableSymbol(pin->get_symbol_from_symbol_table());
      return sb::buildAddressOfOp(sb::buildVarRefExp(ps));
    }
  }
  LOG_ERROR() << "No grid parameter found.\n";
  PSAssert(0);
  return NULL;
}

SgExpression *CUDARuntimeBuilder::BuildGridOffset(
    SgExpression *gvref,
    int num_dim,
    SgExpressionPtrList *offset_exprs,
    bool is_kernel,
    bool is_periodic,
    const StencilIndexList *sil) {
  LOG_DEBUG() << "build offset: " << gvref->unparseToString() << "\n";
  /*
    __PSGridGetOffsetND(g, i)
  */
  GridOffsetAttribute *goa = new GridOffsetAttribute(
      num_dim, is_periodic,
      sil, gvref);  
  std::string func_name = "__PSGridGetOffset";
  if (is_periodic) func_name += "Periodic";
  func_name += toString(num_dim) + "D";
  if (is_kernel) func_name += "Dev";
  SgExprListExp *offset_params = sb::buildExprListExp(gvref);
  FOREACH (it, offset_exprs->begin(),
           offset_exprs->end()) {
    si::appendExpression(offset_params,
                         *it);
    goa->AppendIndex(*it);
  }
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(func_name);
  SgFunctionCallExp *offset_fc =
      sb::buildFunctionCallExp(fs, offset_params);
  rose_util::AddASTAttribute<GridOffsetAttribute>(
      offset_fc, goa);
  return offset_fc;
}

SgClassDeclaration *CUDARuntimeBuilder::BuildGridDevType(
    SgClassDeclaration *grid_decl, GridType *gt) {
  /*
    Let X be the user type name, N be the number of dimension,
    
    struct __PSGridNDX_dev {
      PSIndex dim[N];
      [list of struct member arrays];
    };
   */
  string dev_type_name = "___" + gt->type_name() + "_dev";
  SgClassDeclaration *decl =
      sb::buildStructDeclaration(dev_type_name, gs_);
  SgClassDefinition *dev_def = decl->get_definition();

  // Add member dim
  SgType *dim_type = 
      sb::buildArrayType(BuildIndexType2(gs_),
                         sb::buildIntVal(gt->num_dim()));
  si::appendStatement(sb::buildVariableDeclaration("dim", dim_type),
                      dev_def);
                               
  // Add a pointer for each struct member
  SgClassDefinition *def = gt->point_def();
  const SgDeclarationStatementPtrList &members =
      def->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    SgName member_name = vars[0]->get_name();
    SgType *member_type = vars[0]->get_type();
    SgVariableDeclaration *dev_type_member =
        sb::buildVariableDeclaration(member_name,
                                     sb::buildPointerType(member_type));
    si::appendStatement(dev_type_member, dev_def);
  }

  return decl;
}


} // namespace translator
} // namespace physis

