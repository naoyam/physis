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
    int dim[1];        
    void *p;    
  } __PSGrid1D_dev;

  typedef struct {
    int dim[1];            
    float *p;    
  } __PSGrid1DFloat_dev;

  typedef struct {
    int dim[1];           
    double *p;    
  } __PSGrid1DDouble_dev;

  typedef struct {
    int dim[1];            
    int *p;    
  } __PSGrid1DInt_dev;

  typedef struct {
    int dim[1];            
    long *p;    
  } __PSGrid1DLong_dev;

  // Note: int may not be enough for dim
  typedef struct {
    int dim[2];        
    void *p;    
  } __PSGrid2D_dev;

  typedef struct {
    int dim[2];
    float *p;    
  } __PSGrid2DFloat_dev;

  typedef struct {
    int dim[2];
    double *p;    
  } __PSGrid2DDouble_dev;

  typedef struct {
    int dim[2];
    int *p;    
  } __PSGrid2DInt_dev;

  typedef struct {
    int dim[2];
    long *p;    
  } __PSGrid2DLong_dev;

  typedef struct {
    int dim[3];    
    void *p;    
  } __PSGrid3D_dev;

  typedef struct {
    int dim[3];    
    float *p;    
  } __PSGrid3DFloat_dev;

  typedef struct {
    int dim[3];
    double *p;    
  } __PSGrid3DDouble_dev;

  typedef struct {
    int dim[3];    
    int *p;    
  } __PSGrid3DInt_dev;

  typedef struct {
    int dim[3];    
    long *p;    
  } __PSGrid3DLong_dev;

  typedef struct {
    int dim[3];    
    void *p;
  } __PSGrid_dev;

  typedef struct {
    void *p; // same as dev->p. used only for grids with primitive point type
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
  typedef __PSGrid *PSGrid1DInt;
  typedef __PSGrid *PSGrid2DInt;
  typedef __PSGrid *PSGrid3DInt;
  typedef __PSGrid *PSGrid1DLong;
  typedef __PSGrid *PSGrid2DLong;
  typedef __PSGrid *PSGrid3DLong;
#define DeclareGrid1D(name, type) typedef __PSGrid *PSGrid1D##name;
#define DeclareGrid2D(name, type) typedef __PSGrid *PSGrid2D##name;
#define DeclareGrid3D(name, type) typedef __PSGrid *PSGrid3D##name;
  //#define PSGridDim(p, d) (((__PSGrid *)(p))->dim[(d)])
#define PSGridDim(p, d) ((p)->dim[(d)])
#define __PSGridDimDev(p, d) ((p)->dim[d])
#else  // PHYSIS_USER
  extern __PSGridDimDev(void *p, int);
#endif // PHYSIS_USER
  
  typedef void * (*__PSGrid_devNewFunc)(int num_dims, PSVectorInt dim);
  extern __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                               __PSGrid_devNewFunc func);
  extern void __PSGridSwap(__PSGrid *g);
  extern void __PSGridMirror(__PSGrid *g);
  extern int __PSGridGetID(__PSGrid *g);
  extern void __PSGridSet(__PSGrid *g, void *buf, ...);
  typedef void (*__PSGrid_devFreeFunc)(void *);
  extern void __PSGridFree(__PSGrid *g, __PSGrid_devFreeFunc func);
  typedef void (*__PSGrid_devCopyinFunc)(void *g, const void *src);
  extern void __PSGridCopyin(void *p, const void *src_array,
                             __PSGrid_devCopyinFunc func);
  typedef void (*__PSGrid_devCopyoutFunc)(void *g, void *dst);
  extern void __PSGridCopyout(void *p, void *dst_array,
                             __PSGrid_devCopyoutFunc func);
  
  /** check CUDA error
   * @param[in] message
   */
  extern void __PSCheckCudaError(const char *message);

#ifdef PHYSIS_USER
#define PS_FUNCTION_DEVICE
#else
#define PS_FUNCTION_DEVICE __device__
#endif

  // Forward declartions. These are the only available function
  // signatures for the translator.
  static inline PSIndex __PSGridGetOffset1D(__PSGrid *g, PSIndex i1);
  static inline PSIndex __PSGridGetOffset2D(__PSGrid *g, PSIndex i1,
                                            PSIndex i2);
  static inline PSIndex __PSGridGetOffset3D(__PSGrid *g, PSIndex i1,
                                            PSIndex i2, PSIndex i3);
  static inline PSIndex __PSGridGetOffsetPeriodic1D(__PSGrid *g, PSIndex i1);
  static inline PSIndex __PSGridGetOffsetPeriodic2D(__PSGrid *g, PSIndex i1,
                                                    PSIndex i2);
  static inline PSIndex __PSGridGetOffsetPeriodic3D(__PSGrid *g, PSIndex i1,
                                                    PSIndex i2, PSIndex i3);
  CUDA_DEVICE
  static inline PSIndex __PSGridGetOffset1DDev(const void *g,
                                               PSIndex i1);
  CUDA_DEVICE
  static inline PSIndex __PSGridGetOffset2DDev(const void *g,
                                               PSIndex i1,
                                               PSIndex i2);
  CUDA_DEVICE
  static inline PSIndex __PSGridGetOffset3DDev(const void *g,
                                               PSIndex i1,
                                               PSIndex i2,
                                               PSIndex i3);
  CUDA_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic1DDev(const void *g,
                                                       PSIndex i1);
  CUDA_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic2DDev(const void *g,
                                                       PSIndex i1,
                                                       PSIndex i2);
  static inline PSIndex __PSGridGetOffsetPeriodic3DDev(const void *g,
                                                       PSIndex i1,
                                                       PSIndex i2,
                                                       PSIndex i3);
  // These are only available for generated codes
#ifndef PHYSIS_USER
  static inline PSIndex __PSGridGetOffset1D(__PSGrid *g, PSIndex i1) {
    return i1;
  }
  static inline PSIndex __PSGridGetOffset2D(__PSGrid *g, PSIndex i1,
                                     PSIndex i2) {
    return i1 + i2 * PSGridDim(g, 0);
  }
  static inline PSIndex __PSGridGetOffset3D(__PSGrid *g, PSIndex i1,
                                     PSIndex i2, PSIndex i3) {
    return i1 + i2 * PSGridDim(g, 0) + i3 * PSGridDim(g, 0)
        * PSGridDim(g, 1);
  }

  static inline PSIndex __PSGridGetOffsetPeriodic1D(__PSGrid *g, PSIndex i1) {
    return (i1 + PSGridDim(g, 0)) % PSGridDim(g, 0);
  }
  static inline PSIndex __PSGridGetOffsetPeriodic2D(__PSGrid *g, PSIndex i1,
                                             PSIndex i2) {
    return __PSGridGetOffsetPeriodic1D(g, i1) +
        (i2 + PSGridDim(g, 1)) % PSGridDim(g, 1) * PSGridDim(g, 0);
  }
  static inline PSIndex __PSGridGetOffsetPeriodic3D(__PSGrid *g, PSIndex i1,
                                             PSIndex i2, PSIndex i3) {
    return __PSGridGetOffsetPeriodic2D(g, i1, i2) +
        (i3 + PSGridDim(g, 2)) % PSGridDim(g, 2) * PSGridDim(g, 0) * PSGridDim(g, 1);
  }
  
  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffset1DDev(const void *g,
                                               PSIndex i1) {
    return i1;
  }
  
  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffset2DDev(const void *g,
                                               PSIndex i1,
                                               PSIndex i2) {
    return i1 + i2 * PSGridDim((__PSGrid_dev *)g, 0);
  }

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffset3DDev(const void *g,
                                               PSIndex i1,
                                               PSIndex i2,
                                               PSIndex i3) {
    return i1 + i2 * PSGridDim((__PSGrid_dev*)g, 0)
        + i3 * PSGridDim((__PSGrid_dev*)g, 0)
        * PSGridDim((__PSGrid_dev*)g, 1);
  }

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic1DDev(const void *g,
                                                       PSIndex i1) {
    return (i1 + PSGridDim((__PSGrid_dev*)g, 0)) % PSGridDim((__PSGrid_dev*)g, 0);    
  }
  
  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic2DDev(const void *g,
                                                       PSIndex i1,
                                                       PSIndex i2) {
    return __PSGridGetOffsetPeriodic1DDev(g, i1) +
        (i2 + PSGridDim((__PSGrid_dev*)g, 1)) % PSGridDim((__PSGrid_dev*)g, 1)
        * PSGridDim((__PSGrid_dev*)g, 0);
  }

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic3DDev(const void *g,
                                                       PSIndex i1,
                                                       PSIndex i2,
                                                       PSIndex i3) {
    return __PSGridGetOffsetPeriodic2DDev(g, i1, i2) +
        (i3 + PSGridDim((__PSGrid_dev*)g, 2)) % PSGridDim((__PSGrid_dev*)g, 2)
        * PSGridDim((__PSGrid_dev*)g, 0) * PSGridDim((__PSGrid_dev*)g, 1);
  }
#endif  
  
  extern void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                                  __PSGrid *g);

  extern void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                                   __PSGrid *g);

  extern void __PSReduceGridInt(void *buf, enum PSReduceOp op,
                                __PSGrid *g);

  extern void __PSReduceGridLong(void *buf, enum PSReduceOp op,
                                 __PSGrid *g);

  // CUDA Runtime APIs. Have signatures here to verify generated
  // ASTs.
#ifdef PHYSIS_USER
  typedef void* cudaStream_t;
  typedef int cudaError_t;
  extern cudaError_t cudaStreamSynchronize(cudaStream_t);
  extern cudaError_t cudaFuncSetCacheConfig(const char* func,
                                            int);
  extern double ceil(double x);
  enum cudaFuncCache {
    cudaFuncCachePreferNone,
    cudaFuncCachePreferShared,
    cudaFuncCachePreferL1,
    cudaFuncCachePreferEqual
  };
#endif

#ifdef __cplusplus
}
#endif
  

#endif /* PHYSIS_PHYSIS_CUDA_H_ */
