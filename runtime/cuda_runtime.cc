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
  int __PSGridGetID(___PSGrid *g) {
    return 0;
  }

  ___PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                        int double_buffering) {
    ___PSGrid *g = (___PSGrid*)malloc(sizeof(___PSGrid));
    g->elm_size = elm_size;    
    g->num_dims = num_dims;
    PSVectorIntCopy(g->dim, dim);
    g->num_elms = 1;
    int i;
    for (i = 0; i < num_dims; i++) {
      g->num_elms *= dim[i];
    }
    CUDA_SAFE_CALL(cudaMalloc(&g->p0, g->num_elms * g->elm_size));
    if (!g->p0) return INVALID_GRID;
    CUDA_SAFE_CALL(cudaMemset(g->p0, 0, g->num_elms * g->elm_size));

#if ! defined(AUTO_DOUBLE_BUFFERING)
    PSAssert(!double_buffering);
#else
    if (double_buffering) {
      CUDA_SAFE_CALL(cudaMalloc(&g->p1, g->num_elms * g->elm_size));
      if (!g->p1) return INVALID_GRID;
      CUDA_SAFE_CALL(cudaMemset(g->p1, 0, g->num_elms * g->elm_size));      
    } else {
      g->p1 = g->p0;
    }
#endif

    // Set the on-device data. g->dev data will be copied when calling
    // CUDA global functions.
    switch (g->num_dims) {
      case 1:
        g->dev = malloc(sizeof(___PSGrid1D_dev));
        ((___PSGrid1D_dev*)g->dev)->p0 = g->p0;
#ifdef AUTO_DOUBLE_BUFFERING        
        ((___PSGrid1D_dev*)g->dev)->p1 = g->p1;
#endif        
        ((___PSGrid2D_dev*)g->dev)->dim[0] = g->dim[0];
        break;
      case 2:
        g->dev = malloc(sizeof(___PSGrid2D_dev));
        ((___PSGrid2D_dev*)g->dev)->p0 = g->p0;
#ifdef AUTO_DOUBLE_BUFFERING        
        ((___PSGrid1D_dev*)g->dev)->p1 = g->p1;
#endif        
        ((___PSGrid2D_dev*)g->dev)->dim[0] = g->dim[0];
        ((___PSGrid2D_dev*)g->dev)->dim[1] = g->dim[1];        
        break;
      case 3:
        g->dev = malloc(sizeof(___PSGrid3D_dev));
        ((___PSGrid3D_dev*)g->dev)->p0 = g->p0;
#ifdef AUTO_DOUBLE_BUFFERING        
        ((___PSGrid1D_dev*)g->dev)->p1 = g->p1;
#endif        
        ((___PSGrid3D_dev*)g->dev)->dim[0] = g->dim[0];
        ((___PSGrid3D_dev*)g->dev)->dim[1] = g->dim[1];
        ((___PSGrid3D_dev*)g->dev)->dim[2] = g->dim[2];        
        break;
      default:
        LOG_ERROR() << "Unsupported dimension: " << g->num_dims << "\n";
        PSAbort(1);
    }
        
    return g;
  }

  void PSGridFree(void *p) {
    ___PSGrid *g = (___PSGrid *)p;
    if (g->p0) {
      CUDA_SAFE_CALL(cudaFree(g->p0));
    }
#if defined(AUTO_DOUBLE_BUFFERING)    
    if (g->p1 != NULL && g->p0 != g->p1) {
      CUDA_SAFE_CALL(cudaFree(g->p1));
    }
    g->p0 = g->p1 = NULL;
#else
    g->p0 = NULL;
#endif    
  }

  void PSGridCopyin(void *p, const void *src_array) {
    ___PSGrid *g = (___PSGrid *)p;
    CUDA_SAFE_CALL(cudaMemcpy(g->p0, src_array, g->num_elms*g->elm_size,
                              cudaMemcpyHostToDevice));
  }

  void PSGridCopyout(void *p, void *dst_array) {
    ___PSGrid *g = (___PSGrid *)p;
    CUDA_SAFE_CALL(cudaMemcpy(dst_array, g->p0, g->elm_size * g->num_elms,
                              cudaMemcpyDeviceToHost));
  }

  void __PSGridSwap(___PSGrid *g) {
#if defined(AUTO_DOUBLE_BUFFERING)
    std::swap(g->p0, g->p1);
    std::swap(((___PSGrid1D_dev*)g->dev)->p0,
              ((___PSGrid1D_dev*)g->dev)->p1);
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

  void __PSGridSet(___PSGrid *g, void *buf, ...) {
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
    CUDA_SAFE_CALL(cudaMemcpy(((char *)g->p0) + offset, buf, g->elm_size,
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
