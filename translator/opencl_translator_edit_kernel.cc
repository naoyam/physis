#include "translator/opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/SageBuilderEx.h"

namespace sbx = physis::translator::SageBuilderEx;
namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {


// translateKernelDeclaration:
//
// Translate the declaration (and also its body) of the function
// used in device side.
//
void OpenCLTranslator::translateKernelDeclaration(
    SgFunctionDeclaration *node)
{
  SgFunctionDeclaration *node_pos = node;

  // Fixup argument list in function calls
  // If a grid type variable is in argument list, change it to
  // "__PS_ga_dim_x, ... , __PS_ga_buf, __PS_ga_attr"
  {
    // Check function calls in the body
    SgNodePtrList calls =
      NodeQuery::querySubTree(node_pos, V_SgFunctionCallExp);
    FOREACH (it, calls.begin(), calls.end()) {
      SgFunctionCallExp *fcexp = isSgFunctionCallExp(*it);
      std::string fcexpname = fcexp->getAssociatedFunctionSymbol()->get_name().getString();
      PSAssert(fcexp);

      // Get the argument list of the function
      SgExprListExp *argexplist = fcexp->get_args();
      LOG_DEBUG() << "Checking the arguments of function " << fcexpname << "\n";
      // Create new argument list
      SgExprListExp *newargexplist = new_arguments_of_funccall_in_device(argexplist);
      fcexp->set_args(newargexplist);

      // fcexp->set_function(sb::buildFunctionRefExp(gdim_dev));

    } // FOREACH (it, calls.begin(), calls.end())

  }

  // Rename the call to kernel function to __PS_opencl_InDeviceFunc
  {
      // Search function call
      SgNodePtrList calls =
        NodeQuery::querySubTree(node_pos, V_SgFunctionCallExp);
        FOREACH(it, calls.begin(), calls.end()) {
          SgFunctionCallExp *callexp = isSgFunctionCallExp(*it);
          if (!callexp) continue;

          // If it is not a call to device function, continue
          SgFunctionDeclaration *calldec = callexp->getAssociatedFunctionDeclaration();
          if (! tx_->isKernel(calldec)) continue;

          // Rename caller to __PS_opencl_InDeviceFunc
          std::string oldfuncname = calldec->get_name().getString();
          std::string newfuncname = name_new_kernel(oldfuncname);
          LOG_DEBUG() << "Calling to kernel function " << oldfuncname << " found.\n";
          LOG_DEBUG() << "Changing the call to " << newfuncname << ".\n";
          SgName newsgname(newfuncname);
          callexp->set_function(sb::buildFunctionRefExp(newsgname));
        } // FOREACH(it, calls.begin(), calls.end())
  }

  // Rename function name on the declaration in kernel side
  // to __PS_opencl_kernel, for example
  // https://mailman.nersc.gov/pipermail/rose-public/2011-April/000919.html
  //
  {
    std::string oldkernelname = node->get_name().getString();
    std::string newkernelname = name_new_kernel(oldkernelname);

    LOG_DEBUG() << "Changing the name of the function declaration: from " 
      << oldkernelname << " to " << newkernelname << ".\n";
    // Create declaration copy, and rename it
    SgFunctionDeclaration *node_new = isSgFunctionDeclaration(si::copyStatement(node));
    node_new->set_name(newkernelname);

    // Replace the statement, and add the symbol to the scope if missing
    si::replaceStatement(node, node_new);
    SgFunctionSymbol *newkernelsym = global_scope_->lookup_function_symbol(newkernelname, node_new->get_type());
    if (!newkernelsym) {
      newkernelsym = new SgFunctionSymbol(node_new);
      global_scope_->insert_symbol(newkernelname, newkernelsym);
    }
    node_pos = node_new;
  }

  // Rewrite argument list of the function itself
  SgFunctionParameterList *newparams = sb::buildFunctionParameterList();
  SgFunctionParameterList *oldparams = node_pos->get_parameterList();
  SgInitializedNamePtrList &oldargs = oldparams->get_args();

  FOREACH(it, oldargs.begin(), oldargs.end()) {
      SgInitializedName *newarg = isSgInitializedName(*it);

      // handle grid type
      SgType *cur_type = newarg->get_type();
      if (GridType::isGridType(cur_type)) {
        arg_add_grid_type(newparams, newarg);
        continue;
      } // if (GridType::isGridType(cur_type))

      // If the argument is not grid type, add it as it is
      newparams->append_arg(newarg);
  } // FOREACH(it, oldargs.begin(), oldargs.end())

  // And set the new parameter list
  node_pos->set_parameterList(newparams);

  {
    LOG_INFO() << "Adding #ifdef PHYSIS_OPENCL_KERNEL_MODE\n";
    si::attachArbitraryText(
      node_pos,
      "#else /* #ifndef PHYSIS_OPENCL_KERNEL_MODE */\n",
      PreprocessingInfo::before
      );
    si::attachArbitraryText(
      node_pos,
      "#endif /*#ifndef PHYSIS_OPENCL_KERNEL_MODE */\n",
      PreprocessingInfo::after
      );
    si::attachArbitraryText(
      node_pos,
      "#ifndef PHYSIS_OPENCL_KERNEL_MODE\n",
      PreprocessingInfo::after
      );
#if 0
  si::attachArbitraryText(
    src_->get_globalScope(),
    "#endif /* #ifndef PHYSIS_OPENCK_KERNEL_MODE */\n",
    PreprocessingInfo::after
    );
#endif

  }


} // translateKernelDeclaration

} // namespace translator
} // namespace physis
