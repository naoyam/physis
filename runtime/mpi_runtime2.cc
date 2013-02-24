// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)



#include <stdarg.h>
#include <map>
#include <string>

#include "mpi.h"

#include "runtime/mpi_runtime2.h"
#include "runtime/grid_mpi_debug_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_runtime_common.h"

#include "physis/physis_mpi.h"
#include "physis/physis_util.h"

using std::map;
using std::string;

using namespace physis::runtime;
using physis::IndexArray;
using physis::IntArray;
using physis::SizeArray;

namespace physis {
namespace runtime {


} // namespace runtime
} // namespace physis


template <class T>
static T *__PSGridGetAddr(void *g, const IndexArray &indices) {
  GridMPI2 *gm = (GridMPI2*)g;
  return (T*)(gm->GetAddress(indices));
}
template <class T>
static T *__PSGridGetAddr(void *g, PSIndex x) {
  return __PSGridGetAddr<T>(g, IndexArray(x));
}
template <class T>
static T *__PSGridGetAddr(void *g, PSIndex x, PSIndex y) {
  return __PSGridGetAddr<T>(g, IndexArray(x, y));  
}
template <class T>
static T *__PSGridGetAddr(void *g, PSIndex x, PSIndex y,
                          PSIndex z) {
  return __PSGridGetAddr<T>(g, IndexArray(x, y, z));  
}

#ifdef __cplusplus
extern "C" {
#endif


  __PSGridMPI* __PSGridNewMPI2(PSType type, int elm_size, int dim,
                               const PSVectorInt size,
                               int double_buffering,
                               int attr,
                               const PSVectorInt global_offset,
                               const PSVectorInt stencil_offset_min,
                               const PSVectorInt stencil_offset_max) {
    // NOTE: global_offset is not set by the translator. 0 is assumed.
    PSAssert(global_offset == NULL);

    // ensure the grid size is within the global grid space size
    IndexArray gsize = IndexArray(size);
    if (gsize > gs->global_size()) {
      LOG_ERROR() << "Cannot create grids (size: " << gsize
                  << " larger than the grid space ("
                  << gs->global_size() << "\n";
      return NULL;
    }
    return ((Master2*)master)->GridNew(
        type, elm_size, dim, gsize,
        IndexArray(), stencil_offset_min, stencil_offset_max,
        attr);
  }

  void __PSGridSwap(void *p) {
    // Do nothing
    //((GridMPI *)p)->Swap();
    return;
  }
  
  float *__PSGridGetAddrFloat1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<float>(g, x);
  }
  float *__PSGridGetAddrFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<float>(g, x, y);
  }
  float *__PSGridGetAddrFloat3D(__PSGridMPI *g, PSIndex x, PSIndex y, PSIndex z) {
    return __PSGridGetAddr<float>(g, x, y, z);
  }
  double *__PSGridGetAddrDouble1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<double>(g, x);
  }

  double *__PSGridGetAddrDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<double>(g, x, y);
  }
  
  double *__PSGridGetAddrDouble3D(__PSGridMPI *g, PSIndex x, PSIndex y, PSIndex z) {
    return __PSGridGetAddr<double>(g, x, y, z);
  }

  //
  // Get No Halo
  // These routines are in fact the same as the above routines. Should
  // be removed.
  //
  float *__PSGridGetAddrNoHaloFloat1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<float>(g, x);
  }
  double *__PSGridGetAddrNoHaloDouble1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<double>(g, x);
  }
  float *__PSGridGetAddrNoHaloFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<float>(g, x, y);
  }
  double *__PSGridGetAddrNoHaloDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<double>(g, x, y);
  }
  float *__PSGridGetAddrNoHaloFloat3D(__PSGridMPI *g, PSIndex x,
                                PSIndex y, PSIndex z) {
    return __PSGridGetAddr<float>(g, x, y, z);
  }
  double *__PSGridGetAddrNoHaloDouble3D(__PSGridMPI *g, PSIndex x,
                                 PSIndex y, PSIndex z) {
    return __PSGridGetAddr<double>(g, x, y, z);
  }

  //
  // Emit
  //
  float *__PSGridEmitAddrFloat1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<float>(g, x);
  }
  double *__PSGridEmitAddrDouble1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<double>(g, x);
  }
  float *__PSGridEmitAddrFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<float>(g, x, y);
  }
  double *__PSGridEmitAddrDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<double>(g, x, y);
  }
  float *__PSGridEmitAddrFloat3D(__PSGridMPI *g, PSIndex x,
                                 PSIndex y, PSIndex z) {
    return __PSGridGetAddr<float>(g, x, y, z);
  }
  double *__PSGridEmitAddrDouble3D(__PSGridMPI *g, PSIndex x,
                                   PSIndex y, PSIndex z) {
    return __PSGridGetAddr<double>(g, x, y, z);
  }

  void __PSLoadNeighbor(__PSGridMPI *g,
                        const PSVectorInt offset_min,
                        const PSVectorInt offset_max,
                        int diagonal, int reuse, int overlap,
                        int periodic) {
    if (overlap) LOG_WARNING() << "Overlap possible, but not implemented\n";
    GridMPI *gm = (GridMPI*)g;
    gs->LoadNeighbor(gm, IndexArray(offset_min), IndexArray(offset_max),
                     (bool)diagonal, reuse, periodic);
    return;
  }

  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);
  }
  
  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);    
  }

#if 0
  float __PSGridGetFloat(__PSGridMPI *g, ...) {
    va_list args;
    va_start(args, g);
    float v = __PSGridGet<float>(g, args);
    va_end(args);
    return v;
  }

  double __PSGridGetDouble(__PSGridMPI *g, ...) {
    va_list args;
    va_start(args, g);
    double v = __PSGridGet<double>(g, args);
    va_end(args);
    return v;
  }
  
  void __PSGridSet(__PSGridMPI *g, void *buf, ...) {
    GridMPI *gm = (GridMPI*)g;
    int nd = gm->num_dims();
    va_list vl;
    va_start(vl, buf);
    IndexArray index;
    for (int i = 0; i < nd; ++i) {
      index[i] = va_arg(vl, PSIndex);
    }
    va_end(vl);
    master->GridSet(gm, buf, index);
  }
  
  int __PSIsRoot() {
    return pinfo->IsRoot();
  }

#endif  

#ifdef __cplusplus
}
#endif

