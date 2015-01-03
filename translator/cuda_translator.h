// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_

#include "translator/reference_translator.h"
#include "translator/cuda_runtime_builder.h"

namespace physis {
namespace translator {

class CUDATranslator : public ReferenceTranslator {
 public:
  CUDATranslator(const Configuration &config);
  virtual ~CUDATranslator() {}
  
 protected:  
  /** hold __cuda_block_size_struct type */
  SgType *cuda_block_size_type_;
  
  virtual void FixAST();
  virtual void FixGridType();

  virtual CUDABuilderInterface *builder() {
    return dynamic_cast<CUDABuilderInterface*>(rt_builder_);
  }
  
 public:

  virtual void SetUp(SgProject *project, TranslationContext *context,
                     BuilderInterface *rt_builder);

  virtual void appendNewArgExtra(SgExprListExp *args,
                                 Grid *g,
                                 SgVariableDeclaration *dim_decl);


  virtual void ProcessUserDefinedPointType(SgClassDeclaration *grid_decl,
                                           GridType *gt);

  virtual void TranslateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void TranslateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool is_kernel, bool is_periodic);
  virtual void TranslateGetForUserDefinedType(
      SgDotExp *node, SgPntrArrRefExp *array_top);
  virtual void TranslateEmit(SgFunctionCallExp *node,
                             GridEmitAttribute *attr);
  virtual void TranslateFree(SgFunctionCallExp *node,
                             GridType *gt);
  virtual void TranslateCopyin(SgFunctionCallExp *node,
                               GridType *gt);
  virtual void TranslateCopyout(SgFunctionCallExp *node,
                                GridType *gt);

  virtual void Visit(SgExpression *node);


  /** add dynamic parameter
   * @param[in/out] parlist ... parameter list
   */
  virtual void AddDynamicParameter(SgFunctionParameterList *parlist);
  /** add dynamic argument
   * @param[in/out] args ... arguments
   * @param[in] a_exp ... index expression
   */
  virtual void AddDynamicArgument(SgExprListExp *args, SgExpression *a_exp);
  /** add some code after dlclose()
   * @param[in] scope
   */
  virtual void AddSyncAfterDlclose(SgScopeStatement *scope);

};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_ */
