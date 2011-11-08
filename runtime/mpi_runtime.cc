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

using std::map;
using std::string;

using namespace physis::runtime;

namespace physis {
namespace runtime {

ProcInfo *pinfo;
Master *master;
Client *client;
GridSpaceMPI *gs;

__PSStencilRunClientFunction *__PS_stencils;

} // namespace runtime
} // namespace physis

static ssize_t __PSGridCalcOffset3D(ssize_t x, ssize_t y,
                                    ssize_t z, const IntArray &base,
                                    const IntArray &size) {
  return (x - base[0]) + (y - base[1]) * size[0] +
      (z - base[2]) * size[0] * size[1];
}

static ssize_t __PSGridCalcOffset3D(ssize_t x, ssize_t y, ssize_t z, 
                                    const IntArray &size)
    __attribute__ ((unused));
static ssize_t __PSGridCalcOffset3D(ssize_t x, ssize_t y, ssize_t z, 
                                    const IntArray &size) {
  return x + y * size[0] + z * size[0] * size[1];
}  

static ssize_t __PSGridCalcOffset3D(ssize_t x, ssize_t y, ssize_t z, 
                                    ssize_t xsize, ssize_t ysize)
    __attribute__ ((unused));
static ssize_t __PSGridCalcOffset3D(ssize_t x, ssize_t y, ssize_t z, 
                                    ssize_t xsize, ssize_t ysize) {
  return x + y * xsize + z * xsize * ysize;
}  

static ssize_t __PSGridCalcOffset3D(const IntArray &index,
                                    const IntArray &size) {
  return index[0] + index[1] * size[0] + index[2] * size[0] * size[1];
}

// REFACTORING: This should be a member of GridMPI class
template <class T>
static T *__PSGridGetAddr(GridMPI *gm, IntArray indices) {
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
      ssize_t offset;
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
T *__PSGridEmitAddr3D(__PSGridMPI *g, ssize_t x, ssize_t y,
                         ssize_t z) {
  GridMPI *gm = (GridMPI*)g;
  ssize_t off = __PSGridCalcOffset3D(x, y, z,
                                     gm->local_offset(),
                                     gm->local_size());
  return ((T*)(gm->_data_emit())) + off;
}    

template <class T>
T *__PSGridGetAddrNoHalo3D(__PSGridMPI *g, ssize_t x, ssize_t y,
                              ssize_t z) {
  GridMPI *gm = (GridMPI*)g;
  ssize_t off = __PSGridCalcOffset3D(x, y, z,
                                     gm->local_offset(),
                                     gm->local_size());
  return ((T*)(gm->_data_emit())) + off;
}    

template <class T>
T __PSGridGet(__PSGridMPI *g, va_list args) {
  GridMPI *gm = (GridMPI*)g;
  int nd = gm->num_dims();
  IntArray index;
  for (int i = 0; i < nd; ++i) {
    index[i] = va_arg(args, index_t);
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
    int rank;
    int num_procs;
    // by default the number of dimensions of processes and grids is
    // the same 
    int proc_num_dims;
    va_list vl;
    //const int *grid_size;
    IntArray grid_size;

    physis::runtime::PSInitCommon(argc, argv);
            
    va_start(vl, grid_num_dims);
    for (int i = 0; i < grid_num_dims; ++i) {
      grid_size[i] = va_arg(vl, int);
    }
    int num_stencil_run_calls = va_arg(vl, int);
    __PSStencilRunClientFunction *stencil_funcs
        = va_arg(vl, __PSStencilRunClientFunction*);
    va_end(vl);
    
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    pinfo = new ProcInfo(rank, num_procs);
    LOG_INFO() << *pinfo << "\n";

    IntArray proc_size;
    proc_size.assign(1);
    proc_num_dims = GetProcessDim(argc, argv, proc_size);
    if (proc_num_dims < 0) {
      proc_num_dims = grid_num_dims;
      // 1-D process decomposition by default
      LOG_INFO() <<
          "No process dimension specified; defaulting to 1D decomposition\n";
      proc_size[proc_num_dims-1] = num_procs;
    }

    LOG_INFO() << "Process size: " << proc_size << "\n";
    
    gs = new GridSpaceMPI(grid_num_dims, grid_size,
                          proc_num_dims, proc_size, rank);

    LOG_INFO() << "Grid space: " << *gs << "\n";

    // Set the stencil client functions
    __PS_stencils =
        (__PSStencilRunClientFunction*)malloc(sizeof(__PSStencilRunClientFunction)
                                             * num_stencil_run_calls);
    memcpy(__PS_stencils, stencil_funcs,
           sizeof(__PSStencilRunClientFunction) * num_stencil_run_calls);
    if (rank != 0) {
      LOG_DEBUG() << "I'm a client.\n";
      client = new Client(*pinfo, gs, MPI_COMM_WORLD);
      client->Listen();
      master = NULL;
    } else {
      LOG_DEBUG() << "I'm the master.\n";        
      master = new Master(*pinfo, gs, MPI_COMM_WORLD);
      client = NULL;
    }
    
  }

  void PSFinalize() {
    master->Finalize();
  }

  PSDomain1D PSDomain1DNew(index_t minx, index_t maxx) {
    IntArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IntArray(minx));
    IntArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(maxx));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.assign(0);
      local_max.assign(0);
    }
    PSDomain1D d = {{minx}, {maxx}, {local_min[0]}, {local_max[0]}};
    return d;
  }
  
  PSDomain2D PSDomain2DNew(index_t minx, index_t maxx,
                           index_t miny, index_t maxy) {
    IntArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IntArray(minx, miny));    
    IntArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(maxx, maxy));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.assign(0);
      local_max.assign(0);
    }
    PSDomain2D d = {{minx, miny}, {maxx, maxy},
                    {local_min[0], local_min[1]},
                    {local_max[0], local_max[1]}};
    return d;
  }

  PSDomain3D PSDomain3DNew(index_t minx, index_t maxx,
                           index_t miny, index_t maxy,
                           index_t minz, index_t maxz) {
    IntArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IntArray(minx, miny, minz));        
    IntArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(maxx, maxy, maxz));    
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.assign(0);
      local_max.assign(0);
    }
    PSDomain3D d = {{minx, miny, minz}, {maxx, maxy, maxz},
                    {local_min[0], local_min[1], local_min[2]},
                    {local_max[0], local_max[1], local_max[2]}};
    return d;
  }

  void __PSDomainSetLocalSize(__PSDomain *dom) {
    IntArray local_min = gs->my_offset();
    IntArray global_min(dom->min);
    local_min.SetNoLessThan(global_min);
    IntArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(dom->max));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.assign(0);
      local_max.assign(0);
    }
    local_min.Set(dom->local_min);
    local_max.Set(dom->local_max);
  }
  
  void PSPrintInternalInfo(FILE *out) {
    std::ostringstream ss;
    ss << *pinfo << "\n";
    ss << *gs << "\n";
    fprintf(out, "%s", ss.str().c_str());
  }

  __PSGridMPI* __PSGridNewMPI(PSType type, int elm_size, int dim,
                              const PSVectorInt size,
                              int double_buffering,
                              const PSVectorInt global_offset,
                              int attr) {
    // NOTE: global_offset is not set by the translator. 0 is assumed.
    PSAssert(global_offset == NULL);

    // ensure the grid size is within the global grid space size
    IntArray gsize = IntArray(size);
    if (gsize > gs->global_size()) {
      LOG_ERROR() << "Cannot create grids (size: " << gsize
                  << " larger than the grid space ("
                  << gs->global_size() << "\n";
      return NULL;
    }
    return master->GridNew(type, elm_size, dim, gsize,
                           double_buffering, IntArray(), attr);
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
    IntArray index;
    for (int i = 0; i < nd; ++i) {
      index[i] = va_arg(vl, index_t);
    }
    va_end(vl);
    master->GridSet(gm, buf, index);
  }

  index_t PSGridDim(void *p, int d) {
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
    int *stencil_sizes = new int[num_stencils];
    va_list vl;
    va_start(vl, num_stencils);
    for (int i = 0; i < num_stencils; ++i) {
      size_t stencil_size = va_arg(vl, size_t);
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

  void __PSRegisterStencilRunClient(int id, void *fptr) {
    __PS_stencils[id] = (__PSStencilRunClientFunction)fptr;
  }

  float *__PSGridGetAddrFloat1D(__PSGridMPI *g, ssize_t x) {
    return __PSGridGetAddr<float>((GridMPI*)g, IntArray(x));
  }

  float *__PSGridGetAddrFloat2D(__PSGridMPI *g, ssize_t x, ssize_t y) {
    return __PSGridGetAddr<float>((GridMPI*)g, IntArray(x, y));
  }
  
  float *__PSGridGetAddrFloat3D(__PSGridMPI *g, ssize_t x, ssize_t y, ssize_t z) {
    return __PSGridGetAddr<float>((GridMPI*)g, IntArray(x, y, z));
  }
  double *__PSGridGetAddrDouble1D(__PSGridMPI *g, ssize_t x) {
    return __PSGridGetAddr<double>((GridMPI*)g, IntArray(x));
  }

  double *__PSGridGetAddrDouble2D(__PSGridMPI *g, ssize_t x, ssize_t y) {
    return __PSGridGetAddr<double>((GridMPI*)g, IntArray(x, y));
  }
  
  double *__PSGridGetAddrDouble3D(__PSGridMPI *g, ssize_t x, ssize_t y, ssize_t z) {
    return __PSGridGetAddr<double>((GridMPI*)g, IntArray(x, y, z));
  }


  //
  // Get No Halo
  //
  float *__PSGridGetAddrNoHaloFloat1D(__PSGridMPI *g, ssize_t x) {
    return __PSGridGetAddrNoHalo3D<float>(g, x, 0, 0);
  }
  double *__PSGridGetAddrNoHaloDouble1D(__PSGridMPI *g, ssize_t x) {
    return __PSGridGetAddrNoHalo3D<double>(g, x, 0, 0);
  }
  
  float *__PSGridGetAddrNoHaloFloat2D(__PSGridMPI *g, ssize_t x, ssize_t y) {
    return __PSGridGetAddrNoHalo3D<float>(g, x, y, 0);
  }
  double *__PSGridGetAddrNoHaloDouble2D(__PSGridMPI *g, ssize_t x, ssize_t y) {
    return __PSGridGetAddrNoHalo3D<double>(g, x, y, 0);
  }
  
  float *__PSGridGetAddrNoHaloFloat3D(__PSGridMPI *g, ssize_t x,
                                ssize_t y, ssize_t z) {
    return __PSGridGetAddrNoHalo3D<float>(g, x, y, z);
  }
  double *__PSGridGetAddrNoHaloDouble3D(__PSGridMPI *g, ssize_t x,
                                 ssize_t y, ssize_t z) {
    return __PSGridGetAddrNoHalo3D<double>(g, x, y, z);
  }

  //
  // Emit
  //
  float *__PSGridEmitAddrFloat1D(__PSGridMPI *g, ssize_t x) {
    return __PSGridEmitAddr3D<float>(g, x, 0, 0);
  }
  double *__PSGridEmitAddrDouble1D(__PSGridMPI *g, ssize_t x) {
    return __PSGridEmitAddr3D<double>(g, x, 0, 0);
  }
  
  float *__PSGridEmitAddrFloat2D(__PSGridMPI *g, ssize_t x, ssize_t y) {
    return __PSGridEmitAddr3D<float>(g, x, y, 0);
  }
  double *__PSGridEmitAddrDouble2D(__PSGridMPI *g, ssize_t x, ssize_t y) {
    return __PSGridEmitAddr3D<double>(g, x, y, 0);
  }
  
  float *__PSGridEmitAddrFloat3D(__PSGridMPI *g, ssize_t x,
                                ssize_t y, ssize_t z) {
    return __PSGridEmitAddr3D<float>(g, x, y, z);
  }
  double *__PSGridEmitAddrDouble3D(__PSGridMPI *g, ssize_t x,
                                 ssize_t y, ssize_t z) {
    return __PSGridEmitAddr3D<double>(g, x, y, z);
  }
  
  void __PSLoadNeighbor(__PSGridMPI *g,
                        const PSVectorInt halo_fw_width,
                        const PSVectorInt halo_bw_width,
                        int diagonal, int reuse, int overlap) {
    if (overlap) LOG_WARNING() << "Overlap possible, but not implemented\n";
    GridMPI *gm = (GridMPI*)g;
    gs->LoadNeighbor(gm, IntArray(halo_fw_width), IntArray(halo_bw_width),
                     (bool)diagonal, reuse);
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
                         int min_dim1, index_t min_offset1,
                         int min_dim2, index_t min_offset2,
                         int max_dim1, index_t max_offset1,
                         int max_dim2, index_t max_offset2,
                         int reuse) {
    PSAssert(min_dim1 == max_dim1);
    PSAssert(min_dim2 == max_dim2);
    int dims[] = {min_dim1, min_dim2};
    LoadSubgrid((GridMPI*)g, gs, dims, IntArray(min_offset1, min_offset2),
                IntArray(max_offset1, max_offset2), reuse);
    return;
  }
  
  void __PSLoadSubgrid3D(__PSGridMPI *g, 
                         int min_dim1, index_t min_offset1,
                         int min_dim2, index_t min_offset2,
                         int min_dim3, index_t min_offset3,
                         int max_dim1, index_t max_offset1,
                         int max_dim2, index_t max_offset2,
                         int max_dim3, index_t max_offset3,
                         int reuse) {
    PSAssert(min_dim1 == max_dim1);
    PSAssert(min_dim2 == max_dim2);
    PSAssert(min_dim3 == max_dim3);
    int dims[] = {min_dim1, min_dim2, min_dim3};
    LoadSubgrid((GridMPI*)g, gs, dims, IntArray(min_offset1, min_offset2, min_offset3),
                IntArray(max_offset1, max_offset2, max_offset3), reuse);
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
  

#ifdef __cplusplus
}
#endif

