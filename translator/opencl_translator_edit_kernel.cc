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
      SgFunctionCallExp *fcexp;
#if DEBUG_FIX_DUP_SGEXP
      fcexp = isSgFunctionCallExp(si::copyExpression(isSgFunctionCallExp(*it)));
#else
      fcexp = isSgFunctionCallExp(*it);
#endif
      std::string fcexpname = fcexp->getAssociatedFunctionSymbol()->get_name().getString();
      PSAssert(fcexp);

      // Get the argument list of the function
#if DEBUG_FIX_DUP_SGEXP
      fcexp = isSgFunctionCallExp(si::copyExpression(isSgFunctionCallExp(*it)));
#endif
      SgExprListExp *argexplist = fcexp->get_args();
      LOG_DEBUG() << "Checking the arguments of function " << fcexpname << "\n";
      // Create new argument list
      SgScopeStatement *scope_func = node_pos->get_scope();
      SgExprListExp *newargexplist = 
        new_arguments_of_funccall_in_device(argexplist, scope_func);
      fcexp = isSgFunctionCallExp(*it);
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
#ifdef DEBUG_FIX_AST_APPEND
      si::appendArg(newparams, newarg);
#else
      newparams->append_arg(newarg);
#endif
  } // FOREACH(it, oldargs.begin(), oldargs.end())

#ifdef DEBUG_FIX_CONSISTENCY
  if (1) {
    // Set the scope of the arguments in function definition
    // correctly
    SgInitializedNamePtrList &newargs = newparams->get_args();

    FOREACH(it, newargs.begin(), newargs.end()) {
      SgInitializedName *newarg = isSgInitializedName(*it);
      SgScopeStatement *scope_func = node_pos->get_scope();
      newarg->set_scope(scope_func);
    }
  }
#endif

  // And set the new parameter list
#ifdef DEBUG_FIX_AST_APPEND
  si::setParameterList(node_pos, newparams);
#else
  node_pos->set_parameterList(newparams);
#endif

  {
    LOG_INFO() << "Adding #ifdef << " << kernel_mode_macro() << " \n";
    std::string str_insert = "#else /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/";
    si::attachArbitraryText(
      node_pos,
      str_insert,
      PreprocessingInfo::before
      );
    str_insert = "#endif /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/";
    si::attachArbitraryText(
      node_pos,
      str_insert,
      PreprocessingInfo::after
      );
    str_insert = "#ifndef ";
    str_insert += kernel_mode_macro();
    si::attachArbitraryText(
      node_pos,
      str_insert,
      PreprocessingInfo::after
      );

  }


} // translateKernelDeclaration

} // namespace translator
} // namespace physis
