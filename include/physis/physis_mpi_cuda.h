// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_PHYSIS_MPI_CUDA_H_
#define PHYSIS_PHYSIS_MPI_CUDA_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if !defined(PHYSIS_USER)
#include <mpi.h>
#include <cuda_runtime.h>
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
  extern __PSGridDimDev(const void *p, int);
  extern void dim3(int, ...);
  extern void *__PSGridGetBaseAddr(const void *g);  
#else
  typedef __PSGridMPI *PSGrid1DFloat;
  typedef __PSGridMPI *PSGrid2DFloat;
  typedef __PSGridMPI *PSGrid3DFloat;
  typedef __PSGridMPI *PSGrid1DDouble;
  typedef __PSGridMPI *PSGrid2DDouble;
  typedef __PSGridMPI *PSGrid3DDouble;
  typedef __PSGridMPI *PSGrid1DInt;
  typedef __PSGridMPI *PSGrid2DInt;
  typedef __PSGridMPI *PSGrid3DInt;
  typedef __PSGridMPI *PSGrid1DLong;
  typedef __PSGridMPI *PSGrid2DLong;
  typedef __PSGridMPI *PSGrid3DLong;
  extern PSIndex PSGridDim(void *p, int d);
  extern cudaStream_t stream_inner;
  extern cudaStream_t stream_boundary_copy;
  extern int num_stream_boundary_kernel;
  extern cudaStream_t stream_boundary_kernel[];
#define __PSGridDimDev(p, d) ((p)->dim[d])
#define __PSGridGetBaseAddr(p) ((p)->p0)  
#endif

#define DEFINE_GRID_DEV_GENERIC_TYPE(DIM)        \
  typedef struct {                               \
    int dim[DIM];                                \
    int local_size[DIM];                         \
    int local_offset[DIM];                       \
    void *p;                                     \
  } __PSGrid##DIM##D_dev;             

#define DEFINE_GRID_DEV_TYPE(DIM, TY, TY_NAME)   \
  typedef struct {                               \
    int dim[DIM];                                \
    int local_size[DIM];                         \
    int local_offset[DIM];                       \
    TY *p;                                       \
  } __PSGrid##DIM##D##TY_NAME##_dev;             

  DEFINE_GRID_DEV_GENERIC_TYPE(1);
  DEFINE_GRID_DEV_TYPE(1, float, Float);
  DEFINE_GRID_DEV_TYPE(1, double, Double);  
  DEFINE_GRID_DEV_TYPE(1, int, Int);
  DEFINE_GRID_DEV_TYPE(1, long, Long);  
  DEFINE_GRID_DEV_GENERIC_TYPE(2);
  DEFINE_GRID_DEV_TYPE(2, float, Float);
  DEFINE_GRID_DEV_TYPE(2, double, Double);  
  DEFINE_GRID_DEV_TYPE(2, int, Int);
  DEFINE_GRID_DEV_TYPE(2, long, Long);  
  DEFINE_GRID_DEV_GENERIC_TYPE(3);
  DEFINE_GRID_DEV_TYPE(3, float, Float);
  DEFINE_GRID_DEV_TYPE(3, double, Double);  
  DEFINE_GRID_DEV_TYPE(3, int, Int);
  DEFINE_GRID_DEV_TYPE(3, long, Long);

#define DEFINE_GRID_DEV_TYPE_M(DIM)                \
  typedef struct {                                 \
    int dim[DIM];                                  \
    int local_size[DIM];                           \
    int local_offset[DIM];                         \
    void *p[1];                                    \
  } __PSGrid##DIM##D_dev_m;             
  DEFINE_GRID_DEV_TYPE_M(1);
  DEFINE_GRID_DEV_TYPE_M(2);
  DEFINE_GRID_DEV_TYPE_M(3);


#ifdef __CUDACC__
#define PS_FUNCTION_DEVICE __device__
#else
#define PS_FUNCTION_DEVICE 
#endif

#define __PSGridLocalSizeDev(p, d) ((p)->local_size[d])
#define __PSGridLocalOffsetDev(p, d) ((p)->local_offset[d])  

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffset1DDev(const void *g,
                                               PSIndex i1) {
    return i1 - __PSGridLocalOffsetDev((__PSGrid1D_dev*)g, 0);
  }
  
  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffset2DDev(const void *g,
                                               PSIndex i1,
                                               PSIndex i2) {
    return i1 - __PSGridLocalOffsetDev((__PSGrid2D_dev*)g, 0)
        + (i2 - __PSGridLocalOffsetDev((__PSGrid2D_dev*)g, 1))
        * __PSGridLocalSizeDev((__PSGrid2D_dev *)g, 0);
  }

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffset3DDev(const void *g,
                                               PSIndex i1,
                                               PSIndex i2,
                                               PSIndex i3) {
    return i1 - __PSGridLocalOffsetDev((__PSGrid3D_dev*)g, 0)
        + (i2 - __PSGridLocalOffsetDev((__PSGrid3D_dev*)g, 1))
        * __PSGridLocalSizeDev((__PSGrid3D_dev *)g, 0)
        + (i3 - __PSGridLocalOffsetDev((__PSGrid3D_dev*)g, 2))
        * __PSGridLocalSizeDev((__PSGrid3D_dev *)g, 0)
        * __PSGridLocalSizeDev((__PSGrid3D_dev *)g, 1);
  }

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic1DDev(const void *g,
                                                       PSIndex i1) {
    return (i1 - __PSGridLocalOffsetDev((__PSGrid1D_dev*)g, 0))
        % __PSGridLocalSizeDev((__PSGrid1D_dev*)g, 0);
  }
  
  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic2DDev(const void *g,
                                                       PSIndex i1,
                                                       PSIndex i2) {
    return (i1 - __PSGridLocalOffsetDev((__PSGrid2D_dev*)g, 0))
        % __PSGridLocalSizeDev((__PSGrid2D_dev*)g, 0)
        + ((i2 - __PSGridLocalOffsetDev((__PSGrid2D_dev*)g, 1))
           % __PSGridLocalSizeDev((__PSGrid2D_dev*)g, 1))
        * __PSGridLocalSizeDev((__PSGrid2D_dev *)g, 0);
  }

  PS_FUNCTION_DEVICE
  static inline PSIndex __PSGridGetOffsetPeriodic3DDev(const void *g,
                                                       PSIndex i1,
                                                       PSIndex i2,
                                                       PSIndex i3) {
    return __PS_PERIODIC(i1 - __PSGridLocalOffsetDev((__PSGrid3D_dev*)g, 0),
                         __PSGridLocalSizeDev((__PSGrid3D_dev*)g, 0))
        + __PS_PERIODIC(i2 - __PSGridLocalOffsetDev((__PSGrid3D_dev*)g, 1),
                        __PSGridLocalSizeDev((__PSGrid3D_dev*)g, 1))
        * __PSGridLocalSizeDev((__PSGrid3D_dev *)g, 0)
        + __PS_PERIODIC(i3 - __PSGridLocalOffsetDev((__PSGrid3D_dev*)g, 2),
                        __PSGridLocalSizeDev((__PSGrid3D_dev*)g, 2))
        * __PSGridLocalSizeDev((__PSGrid3D_dev *)g, 0)
        * __PSGridLocalSizeDev((__PSGrid3D_dev *)g, 1);
  }

#if ! defined(PHYSIS_RUNTIME) && ! defined(PHYSIS_USER)
#if 0  
  PS_FUNCTION_DEVICE size_t __PSGridCalcOffset3D(int x, int y, int z,
                                                 int pitch, int dimy) {
    return x + y * pitch + z * pitch * dimy;
  }

  PS_FUNCTION_DEVICE float* __PSGridGetAddrNoHaloFloat3D(
      __PSGrid3DFloat_dev *g,
      int x, int y, int z) {
    x -= g->local_offset[0];
    y -= g->local_offset[1];
    z -= g->local_offset[2];
    return g->p + __PSGridCalcOffset3D(
        x, y, z, g->pitch, g->local_size[1]);    
  }

  PS_FUNCTION_DEVICE float* __PSGridGetAddrNoHaloFloat3DLocal(
      __PSGrid3DFloat_dev *g,
      int x, int y, int z) {
    return g->p + __PSGridCalcOffset3D(
        x, y, z, g->pitch, g->local_size[1]);    
  }
  

  PS_FUNCTION_DEVICE float* __PSGridEmitAddrFloat3D(
      __PSGrid3DFloat_dev *g,
      int x, int y, int z) {
    x -= g->local_offset[0];
    y -= g->local_offset[1];
    z -= g->local_offset[2];
    return g->p + __PSGridCalcOffset3D(x, y, z, g->pitch,
                                        g->local_size[1]);    
  }

  // z
  PS_FUNCTION_DEVICE float* __PSGridGetAddrFloat3D_2_fw(
      __PSGrid3DFloat_dev *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[2] < g->local_size[2]) {
      return __PSGridGetAddrNoHaloFloat3DLocal(
          g, indices[0], indices[1], indices[2]);
    } else {
      indices[2] -= g->local_size[2];
      size_t offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->local_size[0], g->local_size[1]);
      return g->halo[2][1] + offset;
    }
  }

  PS_FUNCTION_DEVICE float* __PSGridGetAddrFloat3D_2_bw(
      __PSGrid3DFloat_dev *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[2] >= 0) {
      return __PSGridGetAddrNoHaloFloat3DLocal(
          g, indices[0], indices[1], indices[2]);
    } else {      
      indices[2] += g->halo_width[2][0];
      size_t offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
         g->local_size[0], g->local_size[1]);
      return g->halo[2][0] + offset;
    }
  }

  // y
  PS_FUNCTION_DEVICE float* __PSGridGetAddrFloat3D_1_fw(
      __PSGrid3DFloat_dev *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[1] < g->local_size[1]) {
      if (indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);        
      }
    } else {
      if (g->diag) indices[2] += g->halo_width[2][0];        
      indices[1] -= g->local_size[1];
      size_t offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->local_size[0], g->halo_width[1][1]);
      return g->halo[1][1] + offset;
    }
  }

  PS_FUNCTION_DEVICE float* __PSGridGetAddrFloat3D_1_bw(
      __PSGrid3DFloat_dev *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[1] >= 0) {
      if (indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);        
      }
    } else {
      if (g->diag) indices[2] += g->halo_width[2][0];        
      indices[1] += g->halo_width[1][0];
      size_t offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->local_size[0], g->halo_width[1][0]);          
      return g->halo[1][0] + offset;
    }
  }

  // x
  PS_FUNCTION_DEVICE float* __PSGridGetAddrFloat3D_0_fw(
      __PSGrid3DFloat_dev *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[0] < g->local_size[0]) {
      // not in the halo region of this dimension
      if (indices[1] < g->local_size[1] &&
          indices[1] >= 0 &&
          indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        // must be inside region
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[1] >= g->local_size[1]) {
        return __PSGridGetAddrFloat3D_1_fw(g, x, y, z);
      } else if (indices[1] < 0) {
        return __PSGridGetAddrFloat3D_1_bw(g, x, y, z);        
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);
      }
    } else {
      size_t halo_size1 = g->local_size[1];
      if (g->diag) {
        indices[1] += g->halo_width[1][0];
        indices[2] += g->halo_width[2][0];        
        halo_size1 += g->halo_width[1][0] +
            g->halo_width[1][1];
      }
      indices[0] -= g->local_size[0];
      size_t offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->halo_width[0][1], halo_size1);
      return g->halo[0][1] + offset;
    }
   }
#endif
  
#endif

#if 0  
#if defined(PHYSIS_TRANSLATOR) || defined(PHYSIS_RUNTIME) || defined(PHYSIS_USER)
  extern float* __PSGridGetAddrFloat3D_0_bw(
      __PSGrid3DFloat_dev *g, int x, int y, int z);
#else
  __device__ float* __PSGridGetAddrFloat3D_0_bw(
      __PSGrid3DFloat_dev *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[0] >= 0) { // not in the halo region of this dimension
      if (indices[1] < g->local_size[1] &&
          indices[1] >= 0 &&
          indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        // must be inside region
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[1] >= g->local_size[1]) {
        return __PSGridGetAddrFloat3D_1_fw(g, x, y, z);
      } else if (indices[1] < 0) {
        return __PSGridGetAddrFloat3D_1_bw(g, x, y, z);        
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);
      }
    } else {
      size_t halo_size1 = g->local_size[1];      
      if (g->diag) {
        indices[1] += g->halo_width[1][0];
        indices[2] += g->halo_width[2][0];        
        halo_size1 += g->halo_width[1][0] +
            g->halo_width[1][1];
      }
      indices[0] += g->halo_width[0][0];
      size_t offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->halo_width[0][0], halo_size1);
      return g->halo[0][0] + offset;
    }
  }
#endif  

#if defined(PHYSIS_TRANSLATOR) || defined(PHYSIS_RUNTIME) || defined(PHYSIS_USER)
  extern float* __PSGridGetAddrFloat3D(__PSGrid3DFloat_dev *g,
                                       int x, int y, int z);
#else  
  __device__ float* __PSGridGetAddrFloat3D(__PSGrid3DFloat_dev *g,
                                           int x, int y, int z) {
    int indices[] = {x - g->local_offset[0],
		     y - g->local_offset[1],
		     z - g->local_offset[2]};
    size_t halo_size[3] = {g->local_size[0], g->local_size[1],
                           g->local_size[2]};          
    for (int i = 0; i < 3; ++i) {
      if (indices[i] < 0 || indices[i] >= g->local_size[i]) {
        float *buf;
        if (g->diag) {
          for (int j = i+1; j < 3; ++j) {
            indices[j] += g->halo_width[j][0];
            halo_size[j] += g->halo_width[j][0] +
                            g->halo_width[j][1];
          }
        }
        size_t offset;
        if (indices[i] < 0) {
          indices[i] += g->halo_width[i][0];
          halo_size[i] = g->halo_width[i][0];
          buf = g->halo[i][0];
        } else {
          indices[i] -= g->local_size[i];
          halo_size[i] = g->halo_width[i][1];
          buf = g->halo[i][1];
        }
        offset = __PSGridCalcOffset3D(
				      indices[0], indices[1], indices[2],
				      halo_size[0], halo_size[1]);
        return buf + offset;
      }
    }
    return __PSGridGetAddrNoHaloFloat3D(g, x, y, z);
  }
#endif
#endif

  extern void __PSDomainSetLocalSize(__PSDomain *dom);
  extern __PSGridMPI* __PSGridNewMPI(
      __PSGridTypeInfo *type_info,
      int dim, const PSVectorInt size, int attr,
      const PSVectorInt global_offset,
      const PSVectorInt stencil_offset_min,
      const PSVectorInt stencil_offset_max,
      const int *stencil_offset_min_member,
      const int *stencil_offset_max_member);
  typedef void *(*__PSGrid_devCopyinFunc)(const void *src, size_t num_elms);
  extern void __PSGridCopyin(void *p, const void *src_array,
                             __PSGrid_devCopyinFunc func);
  typedef void (*__PSGrid_devCopyoutFunc)(void *dst, const void *pack,
                                          size_t num_elms);
  extern void __PSGridCopyout(void *p, void *dst_array,
                              __PSGrid_devCopyoutFunc func);
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
  extern void __PSLoadNeighborMember(__PSGridMPI *g,
                                     int member,
                                     const PSVectorInt halo_fw_width,
                                     const PSVectorInt halo_bw_width,
                                     int diagonal, int reuse,
                                     int overlap, int periodic);
  // Load all members with same halo width
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

  extern void *__PSGridGetDev(void *g);
  extern PSIndex __PSGetLocalSize(int dim);
  extern PSIndex __PSGetLocalOffset(int dim);

  extern __PSDomain __PSDomainShrink(__PSDomain *dom, int width);

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
  extern void __PSReduceGridInt(void *buf, enum PSReduceOp op,
				__PSGridMPI *g);
  extern void __PSReduceGridLong(void *buf, enum PSReduceOp op,
				 __PSGridMPI *g);

  // CUDA Runtime APIs. Have signatures here to verify generated
  // ASTs.
#ifdef PHYSIS_USER
  typedef void* cudaStream_t;
  typedef int cudaError_t;
  extern cudaStream_t stream_inner;
  extern cudaStream_t stream_boundary_kernel;  
  extern cudaError_t cudaStreamSynchronize(cudaStream);
  extern cudaError_t cudaFuncSetCacheConfig(const char* func,
                                            int);
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

#endif /* PHYSIS_PHYSIS_MPI_CUDA_H_ */
