// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_PHYSIS_MPI_H_
#define PHYSIS_PHYSIS_MPI_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if !defined(PHYSIS_USER)
#include <mpi.h>
#endif

#include "physis/physis_common.h"

#ifdef __cplusplus
extern "C" {
#endif

  // __PSGridMPI: dummy type for grid objects. We use the different
  // name than the reference implementation so that CPU and GPU
  // versions could coexist in the future implementation.

#ifdef PHYSIS_USER
  typedef struct {
    int dummy;
  } __PSGridMPI;
#else
  typedef void __PSGridMPI;
  typedef __PSGridMPI *PSGrid1DFloat;
  typedef __PSGridMPI *PSGrid2DFloat;
  typedef __PSGridMPI *PSGrid3DFloat;
  typedef __PSGridMPI *PSGrid1DDouble;
  typedef __PSGridMPI *PSGrid2DDouble;
  typedef __PSGridMPI *PSGrid3DDouble;
  extern PSIndex PSGridDim(void *p, int d);
#endif

  extern void __PSDomainSetLocalSize(__PSDomain *dom);
  extern __PSGridMPI* __PSGridNewMPI(PSType type, int elm_size,
                                     int dim,
                                     const PSVectorInt size,
                                     int double_buffering,
                                     int attr,
                                     const PSVectorInt global_offset,
                                     const PSVectorInt stencil_offset_min,
                                     const PSVectorInt stencil_offset_max);
  extern void __PSGridSwap(__PSGridMPI *g);
  extern void __PSGridMirror(__PSGridMPI *g);
  extern int __PSGridGetID(__PSGridMPI *g);
  extern __PSGridMPI *__PSGetGridByID(int id);
  extern void __PSGridSet(__PSGridMPI *g, void *buf, ...);
  extern float __PSGridGetFloat(__PSGridMPI *g, ...);
  extern double __PSGridGetDouble(__PSGridMPI *g, ...);

  extern void __PSStencilRun(int id, int iter, int num_stencils, ...);

  extern int __PSBcast(void *buf, size_t size);

  extern float *__PSGridGetAddrFloat1D(__PSGridMPI *g, PSIndex x);
  extern float *__PSGridGetAddrFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y);
  extern float *__PSGridGetAddrFloat3D(__PSGridMPI *g, PSIndex x, PSIndex y, PSIndex z);
  extern double *__PSGridGetAddrDouble1D(__PSGridMPI *g, PSIndex x);
  extern double *__PSGridGetAddrDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y);
  extern double *__PSGridGetAddrDouble3D(__PSGridMPI *g, PSIndex x, PSIndex y, PSIndex z);
  extern float *__PSGridGetAddrNoHaloFloat1D(__PSGridMPI *g, PSIndex x);
  extern double *__PSGridGetAddrNoHaloDouble1D(__PSGridMPI *g, PSIndex x);
  extern float *__PSGridGetAddrNoHaloFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y);
  extern double *__PSGridGetAddrNoHaloDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y);
  extern float *__PSGridGetAddrNoHaloFloat3D(__PSGridMPI *g, PSIndex x,
                                             PSIndex y, PSIndex z);
  extern double *__PSGridGetAddrNoHaloDouble3D(__PSGridMPI *g, PSIndex x,
                                               PSIndex y, PSIndex z);
  
  extern PSIndex __PSGridGetOffset1D(__PSGridMPI *g, PSIndex i1);
  extern PSIndex __PSGridGetOffset2D(__PSGridMPI *g, PSIndex i1,
                                     PSIndex i2);
  extern PSIndex __PSGridGetOffset3D(__PSGridMPI *g, PSIndex i1,
                                     PSIndex i2, PSIndex i3);
  extern PSIndex __PSGridGetOffsetPeriodic1D(__PSGridMPI *g, PSIndex i1);
  extern PSIndex __PSGridGetOffsetPeriodic2D(__PSGridMPI *g, PSIndex i1,
                                             PSIndex i2);
  extern PSIndex __PSGridGetOffsetPeriodic3D(__PSGridMPI *g, PSIndex i1,
                                             PSIndex i2, PSIndex i3);

  extern void *__PSGridGetBaseAddr(__PSGridMPI *g);
  
  extern float *__PSGridEmitAddrFloat1D(__PSGridMPI *g, PSIndex x);
  extern double *__PSGridEmitAddrDouble1D(__PSGridMPI *g, PSIndex x);
  extern float *__PSGridEmitAddrFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y);
  extern double *__PSGridEmitAddrDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y);
  extern float *__PSGridEmitAddrFloat3D(__PSGridMPI *g, PSIndex x,
                                        PSIndex y, PSIndex z);
  extern double *__PSGridEmitAddrDouble3D(__PSGridMPI *g, PSIndex x,
                                          PSIndex y, PSIndex z);


  
  extern void __PSLoadNeighbor(__PSGridMPI *g,
                               const PSVectorInt offset_min,
                               const PSVectorInt offset_max,
                               int diagonal, int reuse,
                               int overlap, int periodic);
  extern void __PSLoadSubgrid(__PSGridMPI *g, const __PSGridRange *gr,
                              int reuse);
  extern void __PSLoadSubgrid2D(__PSGridMPI *g, 
                                int min_dim1, PSIndex min_offset1,
                                int min_dim2, PSIndex min_offset2,
                                int max_dim1, PSIndex max_offset1,
                                int max_dim2, PSIndex max_offset2,
                                int reuse);
  extern void __PSLoadSubgrid3D(__PSGridMPI *g,
                                int min_dim1, PSIndex min_offset1,
                                int min_dim2, PSIndex min_offset2,
                                int min_dim3, PSIndex min_offset3,
                                int max_dim1, PSIndex max_offset1,
                                int max_dim2, PSIndex max_offset2,
                                int max_dim3, PSIndex max_offset3,
                                int reuse);
  extern void __PSActivateRemoteGrid(__PSGridMPI *g,
                                     int active);
  extern int __PSIsRoot();

  //! Reduces a grid with an operator.
  /*!
    \param buf A pointer to the output buffer.
    \param op A binary operator to reduce elements.
    \param g A grid.
   */
  extern void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                                  __PSGridMPI *g);
  extern void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                                  __PSGridMPI *g);

#ifdef __cplusplus
}
#endif

#endif /* PHYSIS_PHYSIS_MPI_H_ */
