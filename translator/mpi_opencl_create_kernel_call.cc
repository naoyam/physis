#include "translator/SageBuilderEx.h"
#include "translator/translation_context.h"
#include "translator/translation_util.h"

#include "translator/mpi_opencl_translator.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

SgExprListExp *MPIOpenCLTranslator::BuildKernelCallArgList(
    StencilMap *stencil,
    SgExpressionPtrList &index_args) {
  SgClassDefinition *stencil_def = stencil->GetStencilTypeDefinition();

  // Initialize new argument
  SgExprListExp *args = sb::buildExprListExp();

  // (Old) arguments in index_args can be added into new "args"
  // without any modification
  FOREACH(it, index_args.begin(), index_args.end()) {
    si::appendExpression(args, *it);
  }
  
  // append the fields of the stencil type to the argument list  
  SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *var_decl = isSgVariableDeclaration(*it);
    PSAssert(var_decl);
    SgExpression *exp = sb::buildVarRefExp(var_decl);
    SgVariableDefinition *var_def = var_decl->get_definition();
    PSAssert(var_def);
    SgTypedefType *var_type = isSgTypedefType(var_def->get_type());
    if (GridType::isGridType(var_type)) {
      exp = sb::buildAddressOfOp(exp);
      // skip the grid index field
      ++it;
    }
    si::appendExpression(args, exp);
  }
  return args;
}

SgFunctionCallExp *MPIOpenCLTranslator::BuildKernelCall(
    StencilMap *stencil,
    SgExpressionPtrList &index_args) {
  SgExprListExp *args  = BuildKernelCallArgList(stencil, index_args);

  std::string oldkernelname = stencil->getKernel()->get_name().getString();
  std::string newkernelname = oldkernelname;
 
  do {
    const char *oldkernelname_c = oldkernelname.c_str();

    if (strcmp(oldkernelname_c, "kernel")) // No need to rename
      break;
    newkernelname = opencl_trans_->name_new_kernel(oldkernelname);
  } while(0);


  SgFunctionCallExp *func_call;
  func_call =
      sb::buildFunctionCallExp(
#if 0
          rose_util::getFunctionSymbol(stencil->getKernel()),
#else
          sb::buildFunctionRefExp(newkernelname),
#endif
          args
          );
  return func_call;
}


} // namespace translator
} // namespace physis

