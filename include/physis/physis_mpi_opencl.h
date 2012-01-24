#ifndef PHYSIS_PHYSIS_MPI_OPENCL_H_
#define PHYSIS_PHYSIS_MPI_OPENCL_H_

#ifndef PHYSIS_MPI_OPENCL_KERNEL_MODE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if !defined(PHYSIS_USER)
#include <mpi.h>
#include <CL/cl.h>
#endif

#include "physis/physis_common.h"

#ifdef __cplusplus
extern "C" {
#endif

  // __PSGridMPI: dummy type for grid objects. We use the different
  // name than the reference implementation so that CPU and GPU
  // versions could coexist in the future implementation.

  typedef void __PSGridMPI;
  
#ifdef PHYSIS_USER
 extern __PSGridDimDev(void *p, int);
#else
  typedef __PSGridMPI *PSGrid1DFloat;
  typedef __PSGridMPI *PSGrid2DFloat;
  typedef __PSGridMPI *PSGrid3DFloat;
  typedef __PSGridMPI *PSGrid1DDouble;
  typedef __PSGridMPI *PSGrid2DDouble;
  typedef __PSGridMPI *PSGrid3DDouble;
  extern index_t PSGridDim(void *p, int d);
#define __PSGridDimDev(p, d) ((p)->dim[d])
#endif
  extern unsigned int num_clinfo_boundary_kernel;

  typedef struct {
    void *p0;
    int dim[3];
    int local_size[3];
    int local_offset[3];     
    int pitch;
    void *halo[3][2];
    int halo_width[3][2];        
    int diag;    
  } __PSGrid3DDev;

  typedef struct {
    float *p0;
    int dim[3];
    int local_size[3];
    int local_offset[3]; 
    int pitch;
    float *halo[3][2];  
    int halo_width[3][2];    
    int diag;    
  } __PSGrid3DFloatDev;

  typedef struct {
    double *p0;
    int dim[3];
    int local_size[3];
    int local_offset[3]; 
    int pitch;
    double *halo[3][2];
    int halo_width[3][2];
    int diag;
  } __PSGrid3DDoubleDev;

  
#if defined(PHYSIS_TRANSLATOR) || defined(PHYSIS_RUNTIME) || defined(PHYSIS_USER)
  extern float* __PSGridGetAddrFloat3D_0_bw(
      __PSGrid3DFloatDev *g, int x, int y, int z);
  extern float* __PSGridGetAddrFloat3D(__PSGrid3DFloatDev *g,
                                       int x, int y, int z);
#endif  

  extern void __PSDomainSetLocalSize(__PSDomain *dom);  
  extern __PSGridMPI* __PSGridNewMPI(PSType type, int elm_size, int dim,
                                     const PSVectorInt size,
                                     int double_buffering,
                                     int attr,
                                     const PSVectorInt global_offset);
  extern void __PSGridSwap(__PSGridMPI *g);
  extern void __PSGridMirror(__PSGridMPI *g);
  extern int __PSGridGetID(__PSGridMPI *g);
  extern __PSGridMPI *__PSGetGridByID(int id);
  extern void __PSGridSet(__PSGridMPI *g, void *buf, ...);
  extern float __PSGridGetFloat(__PSGridMPI *g, ...);
  extern double __PSGridGetDouble(__PSGridMPI *g, ...);
  

  extern void __PSStencilRun(int id, int iter, int num_stencils, ...);

  extern int __PSBcast(void *buf, size_t size);

  extern void __PSLoadNeighbor(__PSGridMPI *g,
                               const PSVectorInt halo_fw_width,
                               const PSVectorInt halo_bw_width,
                               int diagonal, int reuse,
                               int overlap, int periodic);
  extern void __PSLoadNeighborStage1(__PSGridMPI *g,
                               const PSVectorInt halo_fw_width,
                               const PSVectorInt halo_bw_width,
                               int diagonal, int reuse,
                               int overlap, int periodic);
  extern void __PSLoadNeighborStage2(__PSGridMPI *g,
                               const PSVectorInt halo_fw_width,
                               const PSVectorInt halo_bw_width,
                               int diagonal, int reuse,
                               int overlap, int periodic);
  extern void __PSLoadSubgrid(__PSGridMPI *g, const __PSGridRange *gr,
                              int reuse);
  extern void __PSLoadSubgrid2D(__PSGridMPI *g, 
                                int min_dim1, index_t min_offset1,
                                int min_dim2, index_t min_offset2,
                                int max_dim1, index_t max_offset1,
                                int max_dim2, index_t max_offset2,
                                int reuse);
  extern void __PSLoadSubgrid3D(__PSGridMPI *g,
                                int min_dim1, index_t min_offset1,
                                int min_dim2, index_t min_offset2,
                                int min_dim3, index_t min_offset3,
                                int max_dim1, index_t max_offset1,
                                int max_dim2, index_t max_offset2,
                                int max_dim3, index_t max_offset3,
                                int reuse);
  extern void __PSActivateRemoteGrid(__PSGridMPI *g,
                                     int active);
  extern int __PSIsRoot();

  extern void *__PSGridGetDev(void *g);
  extern index_t __PSGetLocalSize(int dim);
  extern index_t __PSGetLocalOffset(int dim);

  extern __PSDomain __PSDomainShrink(__PSDomain *dom, int width);

enum CL_STREAM_FLAG {
  USE_GENERIC,
  USE_INNER,
  USE_BOUNDARY_COPY,
  USE_BOUNDARY_KERNEL
};

  extern void __PSSetKernel(
      const char *kernelname,
      enum CL_STREAM_FLAG strm_flg, unsigned int strm_num,
      const char *header_path);
  extern void __PSSetKernelArg(unsigned int arg_index, size_t arg_size, const void *arg_val);
  extern void __PSSetKernelArgCLMem(unsigned int arg_index, const void *arg_val);
  extern void __PSSetKernelArg_Grid3DFloat(unsigned int *p_argc, __PSGrid3DFloatDev *g);
  extern void __PSSetKernelArg_Dom(unsigned int *p_argc, __PSDomain *p_dom);
  extern void __PSRunKernel(size_t *globalsize, size_t *localsize);
  extern void __PS_CL_ThreadSynchronize(void);
  

#ifdef __cplusplus
}
#endif

#endif /* #ifndef PHYSIS_MPI_OPENCL_KERNEL_MODE */

#endif /* PHYSIS_PHYSIS_MPI_CUDA_H_ */
