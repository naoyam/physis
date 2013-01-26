#ifndef PHYSIS_PHYSIS_OPENCL_H_
#define PHYSIS_PHYSIS_OPENCL_H_

/* include physis_opencl_kernel.h in kernel .cl file
   otherwise (i.e. but for .cl file) include physis_common.h
*/
#ifndef PHYSIS_OPENCL_KERNEL_MODE
#include "physis/physis_common.h"
#if !defined(PHYSIS_USER)
#include <CL/cl.h>
#endif
#endif /* ifdef PHYSIS_OPENCL_KERNEL_MODE */
/* 
  The following value must be 0 or 1 according to
  whether we use double buffering or not
*/

/* TODO: 
  In other languages (CUDA, ref), num_elms is defined as int64_t.
  However, on 32 bit environ int64_t is long long int and OpenCL does not
  seem to support long long int
*/
#ifdef __x86_64__
typedef int64_t NUM_ELMS_T;
#else
typedef long int NUM_ELMS_T;
#endif


#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {
    void *buf;
    int dim[1];
  } __PSGrid1DDev;

  typedef struct {
    float *buf;
    int dim[1];    
  } __PSGrid1DFloatDev;

  typedef struct {
    double *buf;
    int dim[1];    
  } __PSGrid1DDoubleDev;

  typedef struct {
    void *buf;
    int dim[2];
  } __PSGrid2DDev;

  typedef struct {
    float *buf;
    int dim[2];
  } __PSGrid2DFloatDev;

  typedef struct {
    double *buf;
    int dim[2];
  } __PSGrid2DDoubleDev;

  typedef struct {
    void *buf;
    int dim[3];
  } __PSGrid3DDev;

  typedef struct {
    float *buf;
    int dim[3];
  } __PSGrid3DFloatDev;

  typedef struct {
    double *buf;
    int dim[3];
  } __PSGrid3DDoubleDev;
  
  typedef struct {
    void *buf;    /* buffer */
    PSVectorInt dim;   /* { NUM_x, NUM_y, NUM_z } or so */
    int elm_size;    /* sizeof(float) or so */
    int num_dims;    /* dimension. 2, 3, or so*/
    NUM_ELMS_T num_elms; /* Well, currently num_elms = II_j dim[j] */
    void *devptr; /* pointer to the data used for device */
    int gridattr; /* grid attribute */
  } __PSGrid;

  /* Copied from physis_cuda.h */
#ifndef PHYSIS_USER
  typedef __PSGrid *PSGrid1DFloat;
  typedef __PSGrid *PSGrid2DFloat;
  typedef __PSGrid *PSGrid3DFloat;
  typedef __PSGrid *PSGrid1DDouble;
  typedef __PSGrid *PSGrid2DDouble;
  typedef __PSGrid *PSGrid3DDouble;
#define PSGridDim(p, d) (((__PSGrid *)(p))->dim[(d)])
#define __PSGridDimDev(p, d) ((p)->dim[d])
#else
  extern __PSGridDimDev(void *p, int);
#endif /* ifndef PHYSIS_USER */

#ifndef PHYSIS_OPENCL_KERNEL_MODE /* don't export these functions in kernel */
  extern __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                               int grid_attribute);
  extern int __PSGridGetID(__PSGrid *g);
  extern void __PSGridSet(__PSGrid *g, const void *ptr_val, ...);
  extern void __PSGridSwap(__PSGrid *g);

  extern void __PSSetKernel(const char *kernelname);
  extern void __PSSetKernelArg(unsigned int arg_index, size_t arg_size, const void *arg_val);
  extern void __PSSetKernelArgCLMem(unsigned int arg_index, const void *arg_val);
  extern void __PSRunKernel(size_t *globalsize, size_t *localsize);
#endif /* ifdef PHYSIS_OPENCL_KERNEL_MODE */

#ifdef __cplusplus
} // extern "C" {}
#endif


#endif /* #define PHYSIS_PHYSIS_OPENCL_H_ */
