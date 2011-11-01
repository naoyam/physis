// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_REDUCE_H_
#define PHYSIS_TRANSLATOR_REDUCE_H_

#include "translator/translator_common.h"
#include "translator/grid.h"
#include "physis/physis_util.h"

#define REDUCE_NAME ("PSReduce")

namespace physis {
namespace translator {

class Reduce: public AstAttribute {
 public:
  enum KIND {GRID, KERNEL};  
  Reduce(SgFunctionCallExp *fc);
  virtual ~Reduce();
  static const std::string name;
  AstAttribute *copy();
  SgFunctionCallExp *reduce_call() { return reduce_call_; };
  bool IsGrid() const;
  bool IsKernel() const;
  //! Returns true if a call is to the reduce intrinsic.
  /*!
    \param call A function call.
    \return True if the call is to the reduce intrinsic.
   */
  static bool IsReduce(SgFunctionCallExp *call);
 protected:
  SgFunctionCallExp *reduce_call_;
  KIND kind_;
};


} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_REDUCE_H_ */
