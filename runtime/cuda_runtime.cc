// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include <stdarg.h>

#include <cuda_runtime.h>

#include "runtime/runtime_common.h"
#include "runtime/runtime_cuda.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/cuda_util.h"
#include "runtime/reduce.h"
#include "physis/physis_cuda.h"

using namespace physis::runtime;

namespace {
RuntimeCUDA *rt;
}

#ifdef __cplusplus
extern "C" {
#endif

  void PSInit(int *argc, char ***argv, int grid_num_dims, ...) {
    rt = new RuntimeCUDA();
    va_list vl;
    va_start(vl, grid_num_dims);
    rt->Init(argc, argv, grid_num_dims, vl);
    CUDA_DEVICE_INIT(0);
    CUDA_CHECK_ERROR("CUDA initialization");
    if (!physis::runtime::CheckCudaCapabilities(2, 0)) {
      PSAbort(1);
    }
  }
  
  void PSFinalize() {
    CUDA_SAFE_CALL(cudaThreadExit());
    delete rt;    
  }

  // Id is not used on shared memory 
  int __PSGridGetID(__PSGrid *g) {
    return 0;
  }

  __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                        __PSGrid_devNewFunc func) {
    __PSGrid *g = (__PSGrid*)malloc(sizeof(__PSGrid));
    g->elm_size = elm_size;    
    g->num_dims = num_dims;
    PSVectorIntCopy(g->dim, dim);
    g->num_elms = 1;
    int i;
    for (i = 0; i < num_dims; i++) {
      g->num_elms *= dim[i];
    }
    
    if (func) {
      g->dev = (__PSGrid_dev*)func(num_dims, dim);
    } else {
      void *buf = NULL;
      CUDA_SAFE_CALL(cudaMalloc(&buf, g->num_elms * g->elm_size));
      if (!buf) return INVALID_GRID;
      CUDA_SAFE_CALL(cudaMemset(buf, 0, g->num_elms * g->elm_size));

      // Set the on-device data. g->dev data will be copied when calling
      // CUDA global functions.
      switch (g->num_dims) {
        case 1:
          g->dev = (__PSGrid_dev*)malloc(sizeof(__PSGrid1D_dev));
          ((__PSGrid1D_dev*)g->dev)->p0 = buf;
          ((__PSGrid2D_dev*)g->dev)->dim[0] = g->dim[0];
          break;
        case 2:
          g->dev = (__PSGrid_dev*)malloc(sizeof(__PSGrid2D_dev));
          ((__PSGrid2D_dev*)g->dev)->p0 = buf;
          ((__PSGrid2D_dev*)g->dev)->dim[0] = g->dim[0];
          ((__PSGrid2D_dev*)g->dev)->dim[1] = g->dim[1];        
          break;
        case 3:
          g->dev = (__PSGrid_dev*)malloc(sizeof(__PSGrid3D_dev));
          ((__PSGrid3D_dev*)g->dev)->p0 = buf;
          ((__PSGrid3D_dev*)g->dev)->dim[0] = g->dim[0];
          ((__PSGrid3D_dev*)g->dev)->dim[1] = g->dim[1];
          ((__PSGrid3D_dev*)g->dev)->dim[2] = g->dim[2];        
          break;
        default:
          LOG_ERROR() << "Unsupported dimension: " << g->num_dims << "\n";
          PSAbort(1);
      }
    }
        
    return g;
  }
  
  void __PSGridFree(__PSGrid *g, __PSGrid_devFreeFunc func) {
    if (g->dev) {
      if (func) {
        func(g->dev);
      } else {
        CUDA_SAFE_CALL(cudaFree(g->dev->p0));
        free(g->dev);              
      }
      g->dev = NULL;
    }
  }

  void PSGridCopyin(void *p, const void *src_array) {
    __PSGrid *g = (__PSGrid *)p;
    CUDA_SAFE_CALL(cudaMemcpy(g->dev->p0, src_array, g->num_elms*g->elm_size,
                              cudaMemcpyHostToDevice));
  }

  void PSGridCopyout(void *p, void *dst_array) {
    __PSGrid *g = (__PSGrid *)p;
    CUDA_SAFE_CALL(cudaMemcpy(dst_array, g->dev->p0, g->elm_size * g->num_elms,
                              cudaMemcpyDeviceToHost));
  }

  void __PSGridSwap(__PSGrid *g) {
#if defined(AUTO_DOUBLE_BUFFERING)
    std::swap(g->dev->p0, g->p1);
    std::swap(((__PSGrid1D_dev*)g->dev)->p0,
              ((__PSGrid1D_dev*)g->dev)->p1);
#endif    
  }

  PSDomain1D PSDomain1DNew(PSIndex minx, PSIndex maxx) {
    PSDomain1D d = {{minx}, {maxx}, {minx}, {maxx}};
    return d;
  }
  
  PSDomain2D PSDomain2DNew(PSIndex minx, PSIndex maxx,
                           PSIndex miny, PSIndex maxy) {
    PSDomain2D d = {{minx, miny}, {maxx, maxy},
                    {minx, miny}, {maxx, maxy}};
    return d;
  }

  PSDomain3D PSDomain3DNew(PSIndex minx, PSIndex maxx,
                           PSIndex miny, PSIndex maxy,
                           PSIndex minz, PSIndex maxz) {
    PSDomain3D d = {{minx, miny, minz}, {maxx, maxy, maxz},
                    {minx, miny, minz}, {maxx, maxy, maxz}};
    return d;
  }

  void __PSGridSet(__PSGrid *g, void *buf, ...) {
    int nd = g->num_dims;
    va_list vl;
    va_start(vl, buf);
    PSIndex offset = 0;
    PSIndex base_offset = 1;
    for (int i = 0; i < nd; ++i) {
      PSIndex idx = va_arg(vl, PSIndex);
      offset += idx * base_offset;
      base_offset *= g->dim[i];
    }
    va_end(vl);
    offset *= g->elm_size;
    CUDA_SAFE_CALL(cudaMemcpy(((char *)g->dev->p0) + offset, buf, g->elm_size,
                              cudaMemcpyHostToDevice));
  }

  //! Check CUDA error
  /*!
   * \param message Additional message to display upon errors.
   */
  void __PSCheckCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error) );
      PSAbort(1);
    }
  }

#ifdef __cplusplus
}
#endif
