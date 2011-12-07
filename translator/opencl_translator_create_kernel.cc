#include "translator/opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/SageBuilderEx.h"

namespace sbx = physis::translator::SageBuilderEx;
namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

// BuildRunKernel:
//
// Generate the declaration part of __PSStencilRun_kernel like
// __kernel void __PSStencilRun_kernel1(
//    long int __PS_dom_xmin, long int __PS_dom_xmax,
//    long int __PS_dom_ymin, long int __PS_dom_ymax,
//    long int __PS_dom_zmin, long int __PS_dom_zmax,
//    long int __PS_g_dim_x, long int __PS_g_dim_y, long int __PS_g_dim_z,
//    __global float *__PS_g0_buf,
//    const long int __PS_g0_attr
// ) {}
//
// The body of __PSStencilRUn_kernel will be generated by
// generateRunKernenBody
//
SgFunctionDeclaration *OpenCLTranslator::BuildRunKernel(
    StencilMap *stencil)
{

  // Initialize parameter params
  SgFunctionParameterList *params = sb::buildFunctionParameterList();

  // Stecil structure
  // Checking the structure like
  // struct __PSStencil_kernel1 {
  //  __PSDomain dom;
  //  PSGrid3DFloat g;
  // int __g_index;
  // };
  SgClassDefinition *param_struct_def = stencil->GetStencilTypeDefinition();
  PSAssert(param_struct_def);

  SgInitializedName *grid_arg = NULL;
  SgInitializedName *dom_arg = NULL;

  unsigned int num_gridparm = 0; // number of parameters with grid type


  const SgDeclarationStatementPtrList &members =
      param_struct_def->get_members();

  // Check each Stencil structure elements
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    SgInitializedName *arg = new SgInitializedName(*vars[0]);
    SgType *type = arg->get_type();
    LOG_DEBUG() << "type: " << type->unparseToString() << "\n";

    // domain type
    if (Domain::isDomainType(type)) {
      if (!dom_arg) {
        dom_arg = arg;
      }

      // Add domain related argument list
      int num_dim = stencil->getNumDim();
      arg_add_dom_type(params, num_dim);

      // handling domain type done: continuing
      continue;

    // Next grid type
    } else if (GridType::isGridType(type)) { // if (Domain::isDomainType(type))

      SgType *gt = BuildOnDeviceGridType(tx_->findGridType(type));
      arg->set_type(gt);
      if (!grid_arg) {
        grid_arg = arg;
      }

      // Add grid related argument list
      int num_dim = stencil->getNumDim();
      arg_add_grid_type(params, type, num_dim, num_gridparm);

      // counter++  
      num_gridparm++;

      // skip the grid index
      ++member;
      // handing grid type done: continuing
      continue;

    } // else if (GridType::isGridType(type))
    // For the other types, add such parameters to params without no modification
    params->append_arg(arg);

  } // FOREACH(member, members.begin(), members.end())

  PSAssert(grid_arg);
  PSAssert(dom_arg);

  LOG_INFO() << "Declaring and defining function named "
             << stencil->getRunName() << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(stencil->getRunName(),
                                           sb::buildVoidType(),
                                           params, global_scope_);
  
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));

  // Add "__kernel" modifier to __PSStencilRun_kernel
  SgFunctionModifier &modifier = run_func->get_functionModifier();
  modifier.setOpenclKernel();

  // Set the body of __PSStencilRun_kernel:
  // The body is generated by generateRunKernenBody
  SgBasicBlock *func_body = generateRunKernelBody(stencil, grid_arg, dom_arg);
  run_func->get_definition()->set_body(func_body);

  {
    LOG_INFO() << "Adding #ifdef << " << kernel_mode_macro() << " \n";
    std::string str_insert = "#else /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/";
    si::attachArbitraryText(
      run_func,
      str_insert,
      PreprocessingInfo::before
      );
    str_insert = "#endif /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/";
    si::attachArbitraryText(
      run_func,
      str_insert,
      PreprocessingInfo::after
      );
    str_insert = "#ifndef ";
    str_insert += kernel_mode_macro();
    si::attachArbitraryText(
      run_func,
      str_insert,
      PreprocessingInfo::after
      );
  }

  return run_func;
} // genreateRunKernel


// generareRunKernelBody:
//
// Generate some part of the body of the stencil kernel like:
// {
//   long x = get_global_id(0);
//   long y = get_global_id(1);
//   long z = 0;
//   (This part is written by BuildDomainInclusionCheck)
//   for (z = __PS_dom_zmin; z < __PS_dom_zmax; ++z) {
//     (This part is written by generateKernelCall) 
//   }
// }
//
SgBasicBlock *OpenCLTranslator::generateRunKernelBody(
    StencilMap *stencil,
    SgInitializedName *grid_arg,
    SgInitializedName *dom_arg)
{
  LOG_DEBUG() << "Generating run kernel body\n";

  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  
  SgExpression *domain = sb::buildVarRefExp(dom_arg);

  SgExpressionPtrList index_args;

  int dim = stencil->getNumDim();
  if (dim < 3) {
    LOG_ERROR() << "not supported yet.\n";
  } else if (dim == 3) {
    // Generate a z-loop
    SgVariableDeclaration *loop_index;
    SgStatement *loop_init, *loop_test;
    SgExpression *loop_incr;
    SgBasicBlock *loop_body;
    SgVariableDeclaration *x_index, *y_index, *z_index;
    x_index = sb::buildVariableDeclaration("x", sb::buildLongType(),
                                           NULL, block);
    y_index = sb::buildVariableDeclaration("y", sb::buildLongType(),
                                           NULL, block);
    z_index = sb::buildVariableDeclaration("z", sb::buildLongType(),
                                           NULL, block);
    block->append_statement(x_index);
    block->append_statement(y_index);
    block->append_statement(z_index);

    if (0) {
    } else { // if (0)

      // x = get_global_id(0);
      x_index->reset_initializer(
          sb::buildAssignInitializer(
            BuildFunctionCall(
              "get_global_id", sb::buildIntVal(0)
                )));

      // y = get_global_id(1);
      y_index->reset_initializer(
          sb::buildAssignInitializer(
            BuildFunctionCall(
              "get_global_id", sb::buildIntVal(1)
                )));

      // z = 0; (anyway)
      z_index->reset_initializer(
        sb::buildAssignInitializer(
            sb::buildIntVal(0)
              ));


      // for (z = ...) {...};
      loop_index = z_index;

       // z = __PS_dom_zmin;
      SgExpression *dom_min_z = sb::buildVarRefExp(name_var_dom(dim - 1, 0));
      loop_init = sb::buildAssignStatement(
          sb::buildVarRefExp(loop_index), dom_min_z);

      // z < __PS_dom_zmax;
      SgExpression *dom_max_z = sb::buildVarRefExp(name_var_dom(dim - 1, 1));
      loop_test = sb::buildExprStatement(
          sb::buildLessThanOp(sb::buildVarRefExp(loop_index),
                              dom_max_z));

      // z++;
      loop_incr =
          sb::buildPlusPlusOp(sb::buildVarRefExp(loop_index));
    } // if (0) {} else {} 

    SgVariableDeclaration* t[] = {x_index, y_index};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
    
    // Build domain range check
    block->append_statement(
        BuildDomainInclusionCheck(range_checking_idx, domain));
    
    // Build loop body
    loop_body = sb::buildBasicBlock();

    // Add kernel call
    index_args.push_back(sb::buildVarRefExp(x_index));
    index_args.push_back(sb::buildVarRefExp(y_index));
    index_args.push_back(sb::buildVarRefExp(loop_index));
    SgFunctionCallExp *kernel_call
        = generateKernelCall(stencil, index_args, loop_body);
    loop_body->append_statement(sb::buildExprStatement(kernel_call));

    SgStatement *loop
        = sb::buildForStatement(loop_init, loop_test, loop_incr, loop_body);

    block->append_statement(loop);

  } // else if (dim == 3)

  return block;
} // generateRunKernelBody



// BuildDomainInclusionCheck:
//
// Generate some part of the body of the stencil kernel like:
//  if ((x < __PS_dom_xmin) || (x >= __PS_dom_xmax)|| (y < __PS_dom_ymin) || (y >= __PS_dom_ymax)) 
//    return ;
//
SgIfStmt *OpenCLTranslator::BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgExpression *dom_ref)
{
  SgExpression *test_all = NULL;
  ENUMERATE (dim, index_it, indices.begin(), indices.end()) {
    SgExpression *idx = sb::buildVarRefExp(*index_it);

    // (x < __PS_dom_xmin) || (x >= __PS_dom_xmax)
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(idx, sb::buildVarRefExp(name_var_dom(dim, 0))),
        sb::buildGreaterOrEqualOp(idx, sb::buildVarRefExp(name_var_dom(dim, 1)))
          );
    if (test_all) {
       // (<test_all>) || (<test>)
      test_all = sb::buildOrOp(test_all, test);
    } else {
      test_all = test;
    }
    
  } // ENUMERATE (dim, index_it, indices.begin(), indices.end()

  // if ( <test_all> ) return;
  SgIfStmt *ifstmt =
      sb::buildIfStmt(test_all, sb::buildReturnStmt(), NULL);
  return ifstmt;
} // BuildDOmainInclusionCheck



// generateKernelCall:
//
// Generate some part of the body of the stencil kernel like:
//    __PS_opencl_kernel1(
//      x, y, z,
//      __PS_g0_dim_x, __PS_g0_dim_y, __PS_g0_dim_z,
//      __PS_g0_buf, __PS_g0_attr
//      );
//
SgFunctionCallExp *OpenCLTranslator::generateKernelCall(
    StencilMap *stencil,
    SgExpressionPtrList &index_args,
    SgScopeStatement *containingScope)
{
  // Stencil kernel structure like
  // struct __PSStencil_kernel {
  //  __PSDomain dom;
  //  PSGrid3DFloat g;
  // int __g_index;
  // };
  SgClassDefinition *stencil_def = stencil->GetStencilTypeDefinition();

  // Initialize argument list
  SgExprListExp *args = sb::buildExprListExp();

  // append the fields of the stencil type to the argument list,
  // like "x, y, z"
  FOREACH(it, index_args.begin(), index_args.end()) {
    args->append_expression(*it);
  }

  // grid type counter
  unsigned int num_gridparm = 0;

  // Now check the each structure element
  SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *var_decl = isSgVariableDeclaration(*it);
    PSAssert(var_decl);
    SgExpression *exp = sb::buildVarRefExp(var_decl);
    SgVariableDefinition *var_def = var_decl->get_definition();
    PSAssert(var_def);
    SgTypedefType *var_type = isSgTypedefType(var_def->get_type());

   // Handling grid type
    if (GridType::isGridType(var_type)) {
      std::string newargname;
      int num_dim = stencil->getNumDim();
      int pos_dim = 0;

      // Add "__PS_g0_dim_x, __PS_g0_dim_y, __PS_g0_dim_z"
      // to args
      for (pos_dim = 0; pos_dim < num_dim; pos_dim++) {
        newargname = name_var_grid_dim(num_gridparm, pos_dim);
        exp = sb::buildVarRefExp(newargname);
        args->append_expression(exp);
      } // for (pos_dim = 0; pos_dim < num_dim; pos_dim++)

      // Add "__PS_g0_buf" to args
      newargname = name_var_gridptr(num_gridparm);
      exp = sb::buildVarRefExp(newargname);
      args->append_expression(exp);

      // Add "__PS_g0_attr" to args
      newargname = name_var_gridattr(num_gridparm);
      exp = sb::buildVarRefExp(newargname);
      args->append_expression(exp);

      // skip the grid index field
      ++it;
      // counter++
      ++num_gridparm;
      // handling grid type done: continuing
      continue;
    } // if (GridType::isGridType(var_type))

    // If the structure element is not grid type nor grid index,
    // Add it to arg as it is
    args->append_expression(exp);
  } // FOREACH(it, ++(members.begin()), members.end())

  // Rename kernel name
  std::string oldkernelname = stencil->getKernel()->get_name().getString();
  std::string newkernelname = name_new_kernel(oldkernelname);
  SgName newsgname(newkernelname);
  SgFunctionCallExp *func_call =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp(newsgname), args
          );

  return func_call;
} // generateKernelCall


} // namespace translator
} // namespace physis
