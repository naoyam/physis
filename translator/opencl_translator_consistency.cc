// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/opencl_translator.h"
#include "translator/rose_util.h"

static int count_err = 0;
#define SGGETNAME(node) ((node)->get_name().getString())

namespace physis {
namespace translator {

void OpenCLTranslator::check_consistency(void) {
  check_consistency_variable();
  check_consistency_function_definition();
} // check_consistency

void OpenCLTranslator::check_consistency_variable(void) {

  SgNodePtrList varlist =
      NodeQuery::querySubTree(project_, V_SgInitializedName);
  FOREACH(it, varlist.begin(), varlist.end()) {
    SgInitializedName *var = isSgInitializedName(*it);
    PSAssert(var);
    std::string var_name = SGGETNAME(var);
    // scope
    SgScopeStatement *var_scope = var->get_scope();
    if (!var_scope) {
      count_err++;
      LOG_DEBUG() << count_err << " var " << var_name << " has no scope\n.";
    }
  } // FOREACH(it, varlist.begin(), varlist.end())

} //

void OpenCLTranslator::check_consistency_function_definition(void) {
  SgNodePtrList deflist =
      NodeQuery::querySubTree(project_, V_SgFunctionDefinition);
  FOREACH(it, deflist.begin(), deflist.end()) {
    SgFunctionDefinition *funcdef = isSgFunctionDefinition(*it);
    PSAssert(funcdef);
    std::string funcdef_name = SGGETNAME(funcdef->get_declaration());

    // body
    SgBasicBlock *body = funcdef->get_body();
    SgNode *body_parent = body->get_parent();
    if (!body_parent) {
      count_err++;
      LOG_DEBUG() << count_err << " func " <<
          funcdef_name << 
          " has a body, but the body does not have a parent.\n";
    }
  } // FOREACH(it, deflist_begin(), deflist.end())
} // check_consistency_function_definition

} // namespace translator
} // namespace physis

