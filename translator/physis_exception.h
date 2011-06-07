// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_PHYSIS_EXCEPTION_H_
#define PHYSIS_TRANSLATOR_PHYSIS_EXCEPTION_H_

#include <exception>

namespace physis {
namespace translator {

class PhysisException
    :public std::exception {
  string msg;
 public:
  explicit PhysisException(const string &msg) throw(): msg(msg) {}
  virtual ~PhysisException() throw() {}
  virtual const char* what() const throw() {
    return msg.c_str();
  }
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_PHYSIS_EXCEPTION_H_ */
