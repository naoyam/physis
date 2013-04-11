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
    int dim[1];    
  } __PSGrid1D_dev;

  typedef struct {
    float *p0;    
    int dim[1];        
  } __PSGrid1DFloat_dev;

  typedef struct {
    double *p0;    
    int dim[1];        
  } __PSGrid1DDouble_dev;

  // Note: int may not be enough for dim
  typedef struct {
    void *p0;    
    int dim[2];    
  } __PSGrid2D_dev;

  typedef struct {
    float *p0;    
    int dim[2];    
  } __PSGrid2DFloat_dev;

  typedef struct {
    double *p0;    
    int dim[2];    
  } __PSGriD2DDouble_dev;

  typedef struct {
    void *p0;    
    int dim[3];    
  } __PSGrid3D_dev;

  typedef struct {
    float *p0;    
    int dim[3];    
  } __PSGrid3DFloat_dev;

  typedef struct {
    double *p0;    
    int dim[3];    
  } __PSGrid3DDouble_dev;

  typedef struct {
    void *p0;
    int dim[3];
  } __PSGrid_dev;

  typedef struct {
    //char *p0;
    PSVectorInt dim;    
    int elm_size;
    int num_dims;
    int64_t num_elms;
    __PSGrid_dev *dev;
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
  
  typedef void * (*__PSGrid_devNewFunc)(int num_dims, PSVectorInt dim);
  extern __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                               __PSGrid_devNewFunc func);
  extern void __PSGridSwap(__PSGrid *g);
  extern void __PSGridMirror(__PSGrid *g);
  extern int __PSGridGetID(__PSGrid *g);
  extern void __PSGridSet(__PSGrid *g, void *buf, ...);
  typedef void (*__PSGrid_devFreeFunc)(void *);
  extern void __PSGridFree(__PSGrid *g, __PSGrid_devFreeFunc func);
  /** check CUDA error
   * @param[in] message
   */
  extern void __PSCheckCudaError(const char *message);

#ifdef PHYSIS_USER
#define CUDA_DEVICE
#else
#define CUDA_DEVICE __device__
#endif
  
  inline PSIndex __PSGridGetOffset1D(__PSGrid *g, PSIndex i1) {
    return i1;
  }
  inline PSIndex __PSGridGetOffset2D(__PSGrid *g, PSIndex i1,
                                     PSIndex i2) {
    return i1 + i2 * PSGridDim(g, 0);
  }
  inline PSIndex __PSGridGetOffset3D(__PSGrid *g, PSIndex i1,
                                     PSIndex i2, PSIndex i3) {
    return i1 + i2 * PSGridDim(g, 0) + i3 * PSGridDim(g, 0)
        * PSGridDim(g, 1);
  }

  inline PSIndex __PSGridGetOffsetPeriodic1D(__PSGrid *g, PSIndex i1) {
    return (i1 + PSGridDim(g, 0)) % PSGridDim(g, 0);
  }
  inline PSIndex __PSGridGetOffsetPeriodic2D(__PSGrid *g, PSIndex i1,
                                             PSIndex i2) {
    return __PSGridGetOffsetPeriodic1D(g, i1) +
        (i2 + PSGridDim(g, 1)) % PSGridDim(g, 1) * PSGridDim(g, 0);
  }
  inline PSIndex __PSGridGetOffsetPeriodic3D(__PSGrid *g, PSIndex i1,
                                             PSIndex i2, PSIndex i3) {
    return __PSGridGetOffsetPeriodic2D(g, i1, i2) +
        (i3 + PSGridDim(g, 2)) % PSGridDim(g, 2) * PSGridDim(g, 0) * PSGridDim(g, 1);
  }
  
  CUDA_DEVICE
  inline PSIndex __PSGridGetOffset1DDev(const void *g,
                                        PSIndex i1) {
    return i1;
  }
  
  CUDA_DEVICE
  inline PSIndex __PSGridGetOffset2DDev(const void *g,
                                        PSIndex i1,
                                        PSIndex i2) {
    return i1 + i2 * PSGridDim((__PSGrid_dev *)g, 0);
  }

  CUDA_DEVICE
  inline PSIndex __PSGridGetOffset3DDev(const void *g,
                                        PSIndex i1,
                                        PSIndex i2,
                                        PSIndex i3) {
    return i1 + i2 * PSGridDim((__PSGrid_dev*)g, 0)
        + i3 * PSGridDim((__PSGrid_dev*)g, 0)
        * PSGridDim((__PSGrid_dev*)g, 1);
  }

  CUDA_DEVICE
  inline PSIndex __PSGridGetOffsetPeriodic1DDev(const void *g,
                                                PSIndex i1) {
    return (i1 + PSGridDim((__PSGrid_dev*)g, 0)) % PSGridDim((__PSGrid_dev*)g, 0);    
  }
  
  CUDA_DEVICE
  inline PSIndex __PSGridGetOffsetPeriodic2DDev(const void *g,
                                                PSIndex i1,
                                                PSIndex i2) {
    return __PSGridGetOffsetPeriodic1DDev(g, i1) +
        (i2 + PSGridDim((__PSGrid_dev*)g, 1)) % PSGridDim((__PSGrid_dev*)g, 1)
        * PSGridDim((__PSGrid_dev*)g, 0);
  }

  CUDA_DEVICE
  inline PSIndex __PSGridGetOffsetPeriodic3DDev(const void *g,
                                                PSIndex i1,
                                                PSIndex i2,
                                                PSIndex i3) {
    return __PSGridGetOffsetPeriodic2DDev(g, i1, i2) +
        (i3 + PSGridDim((__PSGrid_dev*)g, 2)) % PSGridDim((__PSGrid_dev*)g, 2)
        * PSGridDim((__PSGrid_dev*)g, 0) * PSGridDim((__PSGrid_dev*)g, 1);
  }
  
  
  extern void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                                  __PSGrid *g);

  extern void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                                   __PSGrid *g);

  // CUDA Runtime APIs. Have signatures here to verify generated
  // ASTs.
#ifdef PHYSIS_USER
  typedef void* cudaStream_t;
  typedef int cudaError_t;
  extern cudaError_t cudaThreadSynchronize(void);
  extern cudaError_t cudaStreamSynchronize(cudaStream);
  extern cudaError_t cudaFuncSetCacheConfig(const char* func,
                                            int);
#endif

#ifdef __cplusplus
}
#endif
  

#endif /* PHYSIS_PHYSIS_CUDA_H_ */
