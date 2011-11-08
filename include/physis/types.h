// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TYPES_H_
#define PHYSIS_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

  typedef int PSType;
  enum PSPrimitiveType {
    PS_INT = 0,
    PS_LONG = 1,
    PS_FLOAT = 2,
    PS_DOUBLE = 3
  };

#ifdef __cplusplus
}
#endif


#endif /* PHYSIS_TYPES_H_ */
