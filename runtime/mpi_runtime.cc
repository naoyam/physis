// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/mpi_runtime.h"

#include <stdarg.h>
#include <map>
#include <string>

#include "mpi.h"

#include "physis/physis_mpi.h"
#include "physis/physis_util.h"
#include "runtime/grid_mpi_debug_util.h"
#include "runtime/mpi_util.h"
#include "runtime/ipc_mpi.h"
#include "runtime/runtime_mpi.h"

using std::map;
using std::string;

using namespace physis::runtime;

namespace physis {
namespace runtime {

Master *master;
GridSpaceMPI *gs;

} // namespace runtime
} // namespace physis

using physis::IndexArray;
using physis::IntArray;
using physis::SizeArray;

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


#ifdef __cplusplus
extern "C" {
#endif


  // Assumes extra arguments. The first argument is the number of
  // dimensions, and each of the remaining ones is the size of
  // respective dimension.
  void PSInit(int *argc, char ***argv, int grid_num_dims, ...) {
    RuntimeMPI *rt = new RuntimeMPI();
    va_list vl;
    va_start(vl, grid_num_dims);
    rt->Init(argc, argv, grid_num_dims, vl);
    gs = rt->gs();
    if (rt->IsMaster()) {
      master = static_cast<Master*>(rt->proc());
    } else {
      rt->Listen();
    }
  }

  void PSFinalize() {
    master->Finalize();
  }

  PSDomain1D PSDomain1DNew(PSIndex minx, PSIndex maxx) {
    IndexArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IndexArray(minx));
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IndexArray(maxx));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.Set(0);
      local_max.Set(0);
    }
    PSDomain1D d = {{minx}, {maxx}, {local_min[0]}, {local_max[0]}};
    return d;
  }
  
  PSDomain2D PSDomain2DNew(PSIndex minx, PSIndex maxx,
                           PSIndex miny, PSIndex maxy) {
    IndexArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IndexArray(minx, miny));    
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IndexArray(maxx, maxy));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.Set(0);
      local_max.Set(0);
    }
    PSDomain2D d = {{minx, miny}, {maxx, maxy},
                    {local_min[0], local_min[1]},
                    {local_max[0], local_max[1]}};
    return d;
  }

  PSDomain3D PSDomain3DNew(PSIndex minx, PSIndex maxx,
                           PSIndex miny, PSIndex maxy,
                           PSIndex minz, PSIndex maxz) {
    IndexArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IndexArray(minx, miny, minz));        
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IndexArray(maxx, maxy, maxz));    
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.Set(0);
      local_max.Set(0);
    }
    PSDomain3D d = {{minx, miny, minz}, {maxx, maxy, maxz},
                    {local_min[0], local_min[1], local_min[2]},
                    {local_max[0], local_max[1], local_max[2]}};
    return d;
  }

  void __PSDomainSetLocalSize(__PSDomain *dom) {
    IndexArray local_min = gs->my_offset();
    IndexArray global_min(dom->min);
    local_min.SetNoLessThan(global_min);
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IndexArray(dom->max));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.Set(0);
      local_max.Set(0);
    }
    local_min.Set(dom->local_min);
    local_max.Set(dom->local_max);
  }
  
  void PSPrintInternalInfo(FILE *out) {
    std::ostringstream ss;
    if (master) ss << *master << "\n";
    ss << *gs << "\n";
    fprintf(out, "%s", ss.str().c_str());
  }

  __PSGridMPI* __PSGridNewMPI(PSType type, int elm_size, int dim,
                              const PSVectorInt size,
                              int double_buffering,
                              int attr,
                              const PSVectorInt global_offset) {
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
    return master->GridNew(type, elm_size, dim, gsize,
                           double_buffering, IndexArray(), attr);
  }

  void __PSGridSwap(void *p) {
    ((GridMPI *)p)->Swap();
  }

  int __PSGridGetID(__PSGridMPI *g) {
    return ((GridMPI *)g)->id();
  }

  __PSGridMPI *__PSGetGridByID(int id) {
    return (__PSGridMPI*)gs->FindGrid(id);
  }

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

  PSIndex PSGridDim(void *p, int d) {
    Grid *g = (Grid *)p;    
    return g->size_[d];
  }

  void PSGridFree(void *p) {
    master->GridDelete((GridMPI*)p);
  }

  void PSGridCopyin(void *g, const void *buf) {
    master->GridCopyin((GridMPI*)g, buf);
    return;
  }

  void PSGridCopyout(void *g, void *buf) {
    master->GridCopyout((GridMPI*)g, buf);;
    return;
  }

  void __PSStencilRun(int id, int iter, int num_stencils, ...) {
    //master->StencilRun(id, stencil_obj_size, stencil_obj, iter);
    void **stencils = new void*[num_stencils];
    unsigned *stencil_sizes = new unsigned[num_stencils];
    va_list vl;
    va_start(vl, num_stencils);
    for (int i = 0; i < num_stencils; ++i) {
      unsigned stencil_size = (unsigned)va_arg(vl, size_t);
      void *sobj = va_arg(vl, void*);
      stencils[i] = sobj;
      stencil_sizes[i] = stencil_size;
    }
    va_end(vl);
    master->StencilRun(id, iter, num_stencils, stencils, stencil_sizes);
    delete[] stencils;
    delete[] stencil_sizes;
    return;
  }

#if 0  
  void __PSRegisterStencilRunClient(int id, void *fptr) {
    __PS_stencils[id] = (__PSStencilRunClientFunction)fptr;
  }
#endif
  float *__PSGridGetAddrFloat1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<float>((GridMPI*)g, IndexArray(x));
  }

  float *__PSGridGetAddrFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<float>((GridMPI*)g, IndexArray(x, y));
  }
  
  float *__PSGridGetAddrFloat3D(__PSGridMPI *g, PSIndex x, PSIndex y, PSIndex z) {
    return __PSGridGetAddr<float>((GridMPI*)g, IndexArray(x, y, z));
  }
  double *__PSGridGetAddrDouble1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddr<double>((GridMPI*)g, IndexArray(x));
  }

  double *__PSGridGetAddrDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddr<double>((GridMPI*)g, IndexArray(x, y));
  }
  
  double *__PSGridGetAddrDouble3D(__PSGridMPI *g, PSIndex x, PSIndex y, PSIndex z) {
    return __PSGridGetAddr<double>((GridMPI*)g, IndexArray(x, y, z));
  }


  //
  // Get No Halo
  //
  float *__PSGridGetAddrNoHaloFloat1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddrNoHalo3D<float>(g, x, 0, 0);
  }
  double *__PSGridGetAddrNoHaloDouble1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridGetAddrNoHalo3D<double>(g, x, 0, 0);
  }
  
  float *__PSGridGetAddrNoHaloFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddrNoHalo3D<float>(g, x, y, 0);
  }
  double *__PSGridGetAddrNoHaloDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridGetAddrNoHalo3D<double>(g, x, y, 0);
  }
  
  float *__PSGridGetAddrNoHaloFloat3D(__PSGridMPI *g, PSIndex x,
                                PSIndex y, PSIndex z) {
    return __PSGridGetAddrNoHalo3D<float>(g, x, y, z);
  }
  double *__PSGridGetAddrNoHaloDouble3D(__PSGridMPI *g, PSIndex x,
                                 PSIndex y, PSIndex z) {
    return __PSGridGetAddrNoHalo3D<double>(g, x, y, z);
  }

  //
  // Emit
  //
  float *__PSGridEmitAddrFloat1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridEmitAddr3D<float>(g, x, 0, 0);
  }
  double *__PSGridEmitAddrDouble1D(__PSGridMPI *g, PSIndex x) {
    return __PSGridEmitAddr3D<double>(g, x, 0, 0);
  }
  
  float *__PSGridEmitAddrFloat2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridEmitAddr3D<float>(g, x, y, 0);
  }
  double *__PSGridEmitAddrDouble2D(__PSGridMPI *g, PSIndex x, PSIndex y) {
    return __PSGridEmitAddr3D<double>(g, x, y, 0);
  }
  
  float *__PSGridEmitAddrFloat3D(__PSGridMPI *g, PSIndex x,
                                PSIndex y, PSIndex z) {
    return __PSGridEmitAddr3D<float>(g, x, y, z);
  }
  double *__PSGridEmitAddrDouble3D(__PSGridMPI *g, PSIndex x,
                                 PSIndex y, PSIndex z) {
    return __PSGridEmitAddr3D<double>(g, x, y, z);
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

  void __PSLoadSubgrid(__PSGridMPI *g, const __PSGridRange *gr,
                       int reuse) {
    // NOTE: This should be very rare. Not sure it should actually be
    // supported either.
    PSAssert(0 && "Not implemented yet");
    return;
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
    return master->IsRoot();
  }

  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);
  }
  
  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);    
  }
  

#ifdef __cplusplus
}
#endif

