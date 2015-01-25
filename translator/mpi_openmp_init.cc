// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/mpi_openmp_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/mpi_runtime_builder.h"

namespace pu = physis::util;
namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

void MPIOpenMPTranslator::translateInit(SgFunctionCallExp *node) {
  LOG_DEBUG() << "Translating Init call\n";

  // Append the number of run calls
  int num_runs = tx_->run_map().size();
  node->append_arg(sb::buildIntVal(num_runs));

  // let the runtime know about the stencil client handlers
  SgFunctionParameterTypeList *client_func_params
      = sb::buildFunctionParameterTypeList
      (sb::buildIntType(),
       sb::buildPointerType(sb::buildPointerType(sb::buildVoidType())));
  SgFunctionType *client_func_type = sb::buildFunctionType(sb::buildVoidType(),
                                                           client_func_params);
  vector<SgExpression*> client_func_exprs;
  client_func_exprs.resize(num_runs, NULL);
  FOREACH (it, tx_->run_map().begin(), tx_->run_map().end()) {
    const Run* run = it->second;
    SgFunctionRefExp *fref = sb::buildFunctionRefExp(run->GetName(),
                                                     client_func_type);
    client_func_exprs[run->id()] = fref;
  }
  SgInitializer *ai = NULL;
  if (client_func_exprs.size()) {
    ai = sb::buildAggregateInitializer(
        sb::buildExprListExp(client_func_exprs));
  } else {
    //ai = sb::buildAssignInitializer(rose_util::buildNULL());
    ai = sb::buildAggregateInitializer(
        sb::buildExprListExp());
  }
  SgBasicBlock *tmp_block = sb::buildBasicBlock();
  SgVariableDeclaration *clients
      = sb::buildVariableDeclaration("stencil_clients",
                                     sb::buildArrayType(sb::buildPointerType(client_func_type)),
                                     ai, tmp_block);
  PSAssert(clients);
  tmp_block->append_statement(clients);
  node->append_arg(sb::buildVarRefExp(clients));

  // Difference between MPITranslator and MPIOpenMPTranslator:
  // Add the division information to the args
  {
    // See mpi_translator.cc: CheckSizes()
    const DomainSet &ds = tx_->domain_map().begin()->second;
    Domain *d = *(ds.begin());
    int maxdim;
    if (d)
      maxdim = d->num_dims();
    else
      maxdim = 3;
    for (int j = 0; j < maxdim; j++) {
      node->append_arg(sb::buildIntVal(division_[j]));
    }
  }

  tmp_block->append_statement(si::copyStatement(getContainingStatement(node)));
  si::replaceStatement(getContainingStatement(node), tmp_block);
  return;

} // translateInit


} // namespace translator
} // namespace physis

