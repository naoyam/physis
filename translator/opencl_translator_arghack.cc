#include "translator/opencl_translator.h"

#include "translator/translation_context.h"

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

// arg_add_dom_type:
// Add "long __PS_dom_xmin, long __PS_dom_xmax" or so to params
//
void OpenCLTranslator::arg_add_dom_type(
    SgFunctionParameterList *params, 
    int num_dim
)
{
  int pos_dim;
  for (pos_dim = 0; pos_dim < num_dim; pos_dim++) {
    std::string newargname;
    SgInitializedName *newarg;
    newargname = name_var_dom(pos_dim, 0); // __PS_dom_?min
    newarg = sb::buildInitializedName(newargname, sb::buildLongType());
#ifdef DEBUG_FIX_AST_APPEND
    si::appendArg(params, newarg);
#else
    params->append_arg(newarg);
#endif
    newargname = name_var_dom(pos_dim, 1); // __PS_dom_?max
    newarg = sb::buildInitializedName(newargname, sb::buildLongType());
#ifdef DEBUG_FIX_AST_APPEND
    si::appendArg(params, newarg);
#else
    params->append_arg(newarg);
#endif
  } // for (pos_dim = 0; pos_dim < num_dim; pos_dim++)
} // arg_add_dom_type


// arg_add_grid_type:
// Add "long __PS_ga_dim_x, ... , __global float * __PS_ga_buf, __PS_ga_attr"
// to params
//
void OpenCLTranslator::arg_add_grid_type(
  SgFunctionParameterList *params,
  std::string name_arg,
  std::string name_type,
  int num_dim
)
{
      std::string newargname;
      SgInitializedName *newarg;

      // Add "long __PS_ga_dim_x, long __PS_ga_dim_y, long __PS_ga_dim_z"
      // to params
      int pos_dim;
      for (pos_dim = 0; pos_dim < num_dim; pos_dim++) {
        newargname = name_var_grid_dim(name_arg, pos_dim);
        newarg = sb::buildInitializedName(newargname, sb::buildLongType());
#ifdef DEBUG_FIX_AST_APPEND
        si::appendArg(params, newarg);
#else
        params->append_arg(newarg);
#endif
      }

      // Add "__global float *__PS_ga_buf" to params
      // Get name "__PS_ga_buf"
      newargname = name_var_gridptr(name_arg);
#if 0
      // build type
      SgType *basetype;
      if (name_type == "float") {
        basetype = sb::buildFloatType();
      } else {
        basetype = sb::buildDoubleType();
      }
      newarg = sb::buildInitializedName(newargname, sb::buildPointerType(basetype));
      // Add "__global"
      SgStorageModifier &modifier = newarg->get_storageModifier();
      modifier.setOpenclGlobal();
#else
      // Ugly hack!!
      std::string type_global_name = "__global ";
      type_global_name += name_type;
      type_global_name += " *";
      // Here type_global_name should be like"__global float *"
      newarg = sb::buildInitializedName(newargname, sb::buildOpaqueType(type_global_name, global_scope_));
#endif
      // Now actually append new argument
#ifdef DEBUG_FIX_AST_APPEND
      si::appendArg(params, newarg);
#else
      params->append_arg(newarg);
#endif

      // Add "long __PS_ga_gridattr" to params
      newargname = name_var_gridattr(name_arg);
      newarg = sb::buildInitializedName(newargname, sb::buildLongType());
#ifdef DEBUG_FIX_AST_APPEND
      si::appendArg(params, newarg);
#else
      params->append_arg(newarg);
#endif

} // arg_add_grid_type

void OpenCLTranslator::arg_add_grid_type(
    SgFunctionParameterList *params,
    SgType *type,
    int num_dim,
    int num_gridparm
)
{
      if (!GridType::isGridType(type))
        PSAbort(1);

      // Get "float" or so here (type of grid) here
      std::string name_type;
      {
        SgType *ty = tx_->findGridType(type)->getElmType();
        if (isSgTypeFloat(ty)) {
           name_type = "float";
        } else if (isSgTypeDouble(ty)) {
          name_type = "double";
        }
      }

      // Set old arg name
      std::string name_arg;
      {
        char tmpbuf[10];
        snprintf(tmpbuf, 10, "%i", num_gridparm);
        name_arg = "g";
        name_arg += tmpbuf;
      }

      arg_add_grid_type(params, name_arg, name_type, num_dim);

} // arg_add_grid_type


void OpenCLTranslator::arg_add_grid_type(
    SgFunctionParameterList *params,
    SgInitializedName *arg
)
{
      SgType *type = arg->get_type();
      if (!GridType::isGridType(type))
        PSAbort(1);

      GridType *grid_type = tx_->findGridType(type);
      // dimension
      int num_dim = grid_type->getNumDim();

      // Get "float" or so here (type of grid) here
      std::string name_type;
      {
        SgType *ty = tx_->findGridType(type)->getElmType();
        if (isSgTypeFloat(ty)) {
           name_type = "float";
        } else if (isSgTypeDouble(ty)) {
          name_type = "double";
        }
      }

      // Get the variable name of arg
      std::string name_arg = arg->get_name().getString();

      arg_add_grid_type(params, name_arg, name_type, num_dim);

} // arg_add_grid_type

// new_arguments_of_funccall_in_device:
// Generate the new argument of function call in device side source
SgExprListExp * 
  OpenCLTranslator::new_arguments_of_funccall_in_device(
    SgExprListExp *argexplist,
    SgScopeStatement *scope
)
{
      SgExprListExp *newargexplist = sb::buildExprListExp();
      SgExpressionPtrList &argexpptr = argexplist->get_expressions();

      {
        std::string gridname;

        FOREACH(it, argexpptr.begin(), argexpptr.end()) {
          std::string argname;
          std::string newargname;
          SgExpression *sgexp = *it;
          // Get variable type
          do {
            SgVarRefExp *varexp;
#ifdef DEBUG_FIX_DUP_SGEXP
            varexp = isSgVarRefExp(si::copyExpression(sgexp));
#else
            varexp = isSgVarRefExp(sgexp);
#endif
            if (!varexp) break;

            // Check if it is grid type variable
            SgType *type = varexp->get_type();
            if (!GridType::isGridType(type)) break;

#ifdef DEBUG_FIX_DUP_SGEXP
            varexp = isSgVarRefExp(si::copyExpression(sgexp));
#endif
            argname = varexp->get_symbol()->get_name().getString();
            LOG_DEBUG() << "Grid type argument " << argname << " found.\n";

            // Fixing up argument list
            // Find the dimension
            GridType *grid_type = tx_->findGridType(type);
            int num_dim = grid_type->getNumDim();

            // Add __PS_ga_dim_x, __PS_ga_dim_y, __PS_ga_dim_z to params
            int pos_dim;
            for (pos_dim = 0; pos_dim < num_dim; pos_dim++) {
              newargname = name_var_grid_dim(argname, pos_dim);
#ifdef DEBUG_FIX_CONSISTENCY
              si::appendExpression(newargexplist, sb::buildOpaqueVarRefExp(newargname, scope));
#else
#ifdef DEBUG_FIX_AST_APPEND
              si::appendExpression(newargexplist, sb::buildVarRefExp(newargname));
#else
              newargexplist->append_expression(sb::buildVarRefExp(newargname));
#endif
#endif
            }
            // Add __PS_ga_buf to params
            newargname = name_var_gridptr(argname);
#ifdef DEBUG_FIX_CONSISTENCY
              si::appendExpression(newargexplist, sb::buildOpaqueVarRefExp(newargname, scope));
#else
#ifdef DEBUG_FIX_AST_APPEND
            si::appendExpression(newargexplist, sb::buildVarRefExp(newargname));
#else
            newargexplist->append_expression(sb::buildVarRefExp(newargname));
#endif
#endif

            // Add __PS_ga_gridattr" to params
            newargname = name_var_gridattr(argname);
#ifdef DEBUG_FIX_CONSISTENCY
              si::appendExpression(newargexplist, sb::buildOpaqueVarRefExp(newargname, scope));
#else
#ifdef DEBUG_FIX_AST_APPEND
            si::appendExpression(newargexplist, sb::buildVarRefExp(newargname));
#else
            newargexplist->append_expression(sb::buildVarRefExp(newargname));
#endif
#endif

            // Fixing up grid type variable argument done: reset
            sgexp = 0;

          } while(0);

            // Add the original expression to new arglist as it is
          if (sgexp) {
#ifdef DEBUG_FIX_AST_APPEND
            si::appendExpression(newargexplist, sgexp);
#else
            newargexplist->append_expression(sgexp);
#endif
          }

        } // FOREACH(it, fcargs.begin(), fcargs.end())
      }

    // ret
    return newargexplist;

} // new_arguments_of_funccall_in_device


} // namespace translator
} // namespace physis
