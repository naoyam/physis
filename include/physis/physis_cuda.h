// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_PHYSIS_CUDA_H_
#define PHYSIS_PHYSIS_CUDA_H_

#include "physis/physis_common.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {
    void *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    void *p1;
#endif    
    int dim[1];
  } __PSGrid1DDev;

  typedef struct {
    float *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    float *p1;
#endif    
    int dim[1];    
  } __PSGrid1DFloatDev;

  typedef struct {
    double *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    double *p1;
#endif        
    int dim[1];    
  } __PSGrid1DDoubleDev;

  // Note: int may not be enough for dim
  typedef struct {
    void *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    void *p1;
#endif        
    int dim[2];
  } __PSGrid2DDev;

  typedef struct {
    float *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    float *p1;
#endif    
    int dim[2];
  } __PSGrid2DFloatDev;

  typedef struct {
    double *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    double *p1;
#endif    
    int dim[2];
  } __PSGrid2DDoubleDev;

  typedef struct {
    void *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    void *p1;
#endif    
    int dim[3];
  } __PSGrid3DDev;

  typedef struct {
    float *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    float *p1;
#endif    
    int dim[3];
  } __PSGrid3DFloatDev;

  typedef struct {
    double *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    double *p1;
#endif    
    int dim[3];
  } __PSGrid3DDoubleDev;

  typedef struct {
    void *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    void *p1;
#endif    
    int dim[3];
  } __PSGridDev;
  
  typedef struct {
    char *p0;
#ifdef AUTO_DOUBLE_BUFFERING    
    char *p1;
#endif
    PSVectorInt dim;    
    int elm_size;
    int num_dims;
    int64_t num_elms;
    void *dev;
  } __PSGrid;

#ifndef PHYSIS_USER
  typedef __PSGrid *PSGrid1DFloat;
  typedef __PSGrid *PSGrid2DFloat;
  typedef __PSGrid *PSGrid3DFloat;
  typedef __PSGrid *PSGrid1DDouble;
  typedef __PSGrid *PSGrid2DDouble;
  typedef __PSGrid *PSGrid3DDouble;
  //#define PSGridDim(p, d) (((__PSGrid *)(p))->dim[(d)])
#define PSGridDim(p, d) ((p)->dim[(d)])
#define __PSGridDimDev(p, d) ((p)->dim[d])
#else
  extern __PSGridDimDev(void *p, int);
#endif

  extern __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                               int double_buffering);
  extern void __PSGridSwap(__PSGrid *g);
  extern void __PSGridMirror(__PSGrid *g);
  extern int __PSGridGetID(__PSGrid *g);
  extern void __PSGridSet(__PSGrid *g, void *buf, ...);

#ifdef PHYSIS_USER
#define CUDA_DEVICE
#else
#define CUDA_DEVICE __device__
#endif
  
  inline PSIndexType __PSGridGetOffset1D(__PSGrid *g, PSIndexType i1) {
    return i1;
  }
  inline PSIndexType __PSGridGetOffset2D(__PSGrid *g, PSIndexType i1,
                                         PSIndexType i2) {
    return i1 + i2 * PSGridDim(g, 0);
  }
  inline PSIndexType __PSGridGetOffset3D(__PSGrid *g, PSIndexType i1,
                                         PSIndexType i2, PSIndexType i3) {
    return i1 + i2 * PSGridDim(g, 0) + i3 * PSGridDim(g, 0)
        * PSGridDim(g, 1);
  }
  
  CUDA_DEVICE
  inline PSIndexType __PSGridGetOffset1DDev(void *g,
                                            PSIndexType i1) {
    return i1;
  }
  
  CUDA_DEVICE
  inline PSIndexType __PSGridGetOffset2DDev(void *g,
                                            PSIndexType i1,
                                            PSIndexType i2) {
    return i1 + i2 * PSGridDim((__PSGridDev *)g, 0);
  }

  CUDA_DEVICE
  inline PSIndexType __PSGridGetOffset3DDev(void *g,
                                            PSIndexType i1,
                                            PSIndexType i2,
                                            PSIndexType i3) {
    return i1 + i2 * PSGridDim((__PSGridDev*)g, 0)
        + i3 * PSGridDim((__PSGridDev*)g, 0)
        * PSGridDim((__PSGridDev*)g, 1);
  }
  
  extern void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                                  __PSGrid *g);

  extern void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                                   __PSGrid *g);


#ifdef __cplusplus
}
#endif
  

#endif /* PHYSIS_PHYSIS_CUDA_H_ */
