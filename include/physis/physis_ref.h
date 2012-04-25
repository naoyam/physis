// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_PHYSIS_REF_H_
#define PHYSIS_PHYSIS_REF_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "physis/physis_common.h"

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef struct {
    int elm_size;
    int num_dims;
    int64_t num_elms;
    PSVectorInt dim;
    void *p0, *p1;
  } __PSGrid;

#ifndef PHYSIS_USER
  typedef __PSGrid *PSGrid1DFloat;
  typedef __PSGrid *PSGrid2DFloat;
  typedef __PSGrid *PSGrid3DFloat;
  typedef __PSGrid *PSGrid1DDouble;
  typedef __PSGrid *PSGrid2DDouble;
  typedef __PSGrid *PSGrid3DDouble;
#define PSGridDim(p, d) (((__PSGrid *)(p))->dim[(d)])  
#endif
  
  extern __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                               int double_buffering);
  extern void __PSGridSwap(__PSGrid *g);
  extern void __PSGridMirror(__PSGrid *g);
  extern int __PSGridGetID(__PSGrid *g);
  extern void __PSGridSet(__PSGrid *g, void *buf, ...);
  extern void __PSGridGet(__PSGrid *g, void *buf, ...);

  inline PSIndexType __PSGridGetOffset1D(__PSGrid *g, PSIndexType i1) {
    return i1;
  }
  inline PSIndexType __PSGridGetOffset2D(__PSGrid *g, PSIndexType i1,
                                         PSIndexType i2) {
    return i1 + i2 * PSGridDim(g, 0);
  }
  inline PSIndexType __PSGridGetOffset3D(__PSGrid *g, PSIndexType i1,
                                         PSIndexType i2, PSIndexType i3) {
    return i1 + i2 * PSGridDim(g, 0) + i3 * PSGridDim(g, 0) * PSGridDim(g, 1);
  }

  inline PSIndexType __PSGridGetOffsetPeriodic1D(__PSGrid *g, PSIndexType i1) {
    return (i1 + PSGridDim(g, 0)) % PSGridDim(g, 0);
  }
  inline PSIndexType __PSGridGetOffsetPeriodic2D(__PSGrid *g, PSIndexType i1,
                                                 PSIndexType i2) {
    return __PSGridGetOffsetPeriodic1D(g, i1) +
        (i2 + PSGridDim(g, 1)) % PSGridDim(g, 1) * PSGridDim(g, 0);
  }
  inline PSIndexType __PSGridGetOffsetPeriodic3D(__PSGrid *g, PSIndexType i1,
                                                 PSIndexType i2, PSIndexType i3) {
    return __PSGridGetOffsetPeriodic2D(g, i1, i2) +
        (i3 + PSGridDim(g, 2)) % PSGridDim(g, 2) * PSGridDim(g, 0) * PSGridDim(g, 1);
  }

  typedef void (*ReducerFunc)();
  
  //extern void __PSReduceGrid(void *buf, __PSGrid *g, ReducerFunc f);
  //! Reduces a grid with an operator.
  /*!
    \param buf A pointer to the output buffer.
    \param op A binary operator to reduce elements.
    \param g A grid.
   */
  extern void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                                  __PSGrid *g);
  extern void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                                  __PSGrid *g);
  
#ifdef __cplusplus
}
#endif

#endif /* PHYSIS_PHYSIS_REF_H_ */
