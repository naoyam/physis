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


#if 0

static PSIndex __PSGridCalcOffset3D(PSIndex x, PSIndex y,
                                    PSIndex z, const IndexArray &base,
                                    const IndexArray &size) {
  return (x - base[0]) + (y - base[1]) * size[0] +
      (z - base[2]) * size[0] * size[1];
}

static PSIndex __PSGridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                    const IndexArray &size)
    __attribute__ ((unused));
static PSIndex __PSGridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                       const IndexArray &size) {
  return x + y * size[0] + z * size[0] * size[1];
}  

static PSIndex __PSGridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                    PSIndex xsize, PSIndex ysize)
    __attribute__ ((unused));
static PSIndex __PSGridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                    PSIndex xsize, PSIndex ysize) {
  return x + y * xsize + z * xsize * ysize;
}  

static PSIndex __PSGridCalcOffset3D(const IndexArray &index,
                                    const IndexArray &size) {
  return index[0] + index[1] * size[0] + index[2] * size[0] * size[1];
}

// REFACTORING: This should be a member of GridMPI class
template <class T>
static T *__PSGridGetAddr(GridMPI *gm, IndexArray indices) {
  // Use the remote grid if remote_grid_active is true.
  if (gm->remote_grid_active()) {
    GridMPI *rmg = gm->remote_grid();
    indices -= rmg->local_offset();
    return ((T*)(rmg->_data())) +
        __PSGridCalcOffset3D(indices, rmg->local_size());
  }
  
  indices -= gm->local_offset();
  bool diag = gm->halo_has_diagonal();
  for (int i = 0; i < PS_MAX_DIM; ++i) {
    if (indices[i] < 0 || indices[i] >= gm->local_size()[i]) {
      for (int j = i+1; j < PS_MAX_DIM; ++j) {
        if (diag) indices[j] += gm->halo_bw_width()[j];
      }
      PSIndex offset;
      T *buf;
      if (indices[i] < 0) {
        indices[i] += gm->halo_bw_width()[i];
        offset = __PSGridCalcOffset3D(indices, gm->halo_bw_size()[i]);
        buf = (T*)gm->_halo_peer_bw()[i];
      } else {
        indices[i] -= gm->local_size()[i];
        offset = __PSGridCalcOffset3D(indices, gm->halo_fw_size()[i]);
        buf = (T*)gm->_halo_peer_fw()[i];
      }
      return buf + offset;
    }
  }

  return ((T*)(gm->_data())) +
      __PSGridCalcOffset3D(indices, gm->local_size());
}

template <class T>
T *__PSGridEmitAddr3D(__PSGridMPI *g, PSIndex x, PSIndex y,
                      PSIndex z) {
  GridMPI *gm = (GridMPI*)g;
  PSIndex off = __PSGridCalcOffset3D(x, y, z,
                                     gm->local_offset(),
                                     gm->local_size());
  return ((T*)(gm->_data_emit())) + off;
}    

template <class T>
T *__PSGridGetAddrNoHalo3D(__PSGridMPI *g, PSIndex x, PSIndex y,
                           PSIndex z) {
  GridMPI *gm = (GridMPI*)g;
  PSIndex off = __PSGridCalcOffset3D(x, y, z,
                                     gm->local_offset(),
                                     gm->local_size());
  return ((T*)(gm->_data_emit())) + off;
}    

template <class T>
T __PSGridGet(__PSGridMPI *g, va_list args) {
  GridMPI *gm = (GridMPI*)g;
  int nd = gm->num_dims();
  IndexArray index;
  for (int i = 0; i < nd; ++i) {
    index[i] = va_arg(args, PSIndex);
  }
  T v;
  master->GridGet(gm, &v, index);
  return v;
}
#endif

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

  void __PSRegisterStencilRunClient(int id, void *fptr) {
    __PS_stencils[id] = (__PSStencilRunClientFunction)fptr;
  }
  

  void __PSLoadSubgrid2D(__PSGridMPI *g, 
                         int min_dim1, PSIndex min_offset1,
                         int min_dim2, PSIndex min_offset2,
                         int max_dim1, PSIndex max_offset1,
                         int max_dim2, PSIndex max_offset2,
                         int reuse) {
    PSAssert(min_dim1 == max_dim1);
    PSAssert(min_dim2 == max_dim2);
    int dims[] = {min_dim1, min_dim2};
    LoadSubgrid((GridMPI*)g, gs, dims, IndexArray(min_offset1, min_offset2),
                IndexArray(max_offset1, max_offset2), reuse);
    return;
  }
  
  void __PSLoadSubgrid3D(__PSGridMPI *g, 
                         int min_dim1, PSIndex min_offset1,
                         int min_dim2, PSIndex min_offset2,
                         int min_dim3, PSIndex min_offset3,
                         int max_dim1, PSIndex max_offset1,
                         int max_dim2, PSIndex max_offset2,
                         int max_dim3, PSIndex max_offset3,
                         int reuse) {
    PSAssert(min_dim1 == max_dim1);
    PSAssert(min_dim2 == max_dim2);
    PSAssert(min_dim3 == max_dim3);
    int dims[] = {min_dim1, min_dim2, min_dim3};
    LoadSubgrid((GridMPI*)g, gs, dims, IndexArray(min_offset1, min_offset2, min_offset3),
                IndexArray(max_offset1, max_offset2, max_offset3), reuse);
    return;
  }

  void __PSActivateRemoteGrid(__PSGridMPI *g, int active) {
    GridMPI *gm = (GridMPI*)g;
    gm->remote_grid_active() = active;
  }

  int __PSIsRoot() {
    return pinfo->IsRoot();
  }

  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);
  }
  
  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);    
  }
#endif  

#ifdef __cplusplus
}
#endif

