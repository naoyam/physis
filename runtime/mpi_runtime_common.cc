// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include <stdarg.h>
#include <map>
#include <string>

#include "runtime/mpi_runtime_common.h"

#include "mpi.h"

#include "physis/physis_mpi.h"
#include "physis/physis_util.h"
#include "runtime/grid_mpi_debug_util.h"
#include "runtime/mpi_util.h"
#include "runtime/grid_mpi2.h"
#include "runtime/rpc_mpi2.h"

using std::map;
using std::string;

using namespace physis::runtime;

using physis::IndexArray;
using physis::IntArray;
using physis::SizeArray;

namespace physis {
namespace runtime {

__PSStencilRunClientFunction *__PS_stencils;

ProcInfo *pinfo;
Master *master;
Client *client;
GridSpaceMPI *gs;

} // namespace runtime
} // namespace physis

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
    IndexArray grid_size;

    physis::runtime::PSInitCommon(argc, argv);
            
    va_start(vl, grid_num_dims);
    for (int i = 0; i < grid_num_dims; ++i) {
      grid_size[i] = va_arg(vl, PSIndex);
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
    proc_size.Set(1);
    proc_num_dims = GetProcessDim(argc, argv, proc_size);
    if (proc_num_dims < 0) {
      proc_num_dims = grid_num_dims;
      // 1-D process decomposition by default
      LOG_INFO() <<
          "No process dimension specified; defaulting to 1D decomposition\n";
      proc_size[proc_num_dims-1] = num_procs;
    }

    LOG_INFO() << "Process size: " << proc_size << "\n";

#ifdef MPI_RUNTIME_2
    gs = new GridSpaceMPI2(grid_num_dims, grid_size,
                           proc_num_dims, proc_size, rank);
#else    
    gs = new GridSpaceMPI(grid_num_dims, grid_size,
                          proc_num_dims, proc_size, rank);
#endif    

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
#ifdef MPI_RUNTIME_2
      master = new Master2(*pinfo, gs, MPI_COMM_WORLD);
#else
      master = new Master(*pinfo, gs, MPI_COMM_WORLD);
#endif
      client = NULL;
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
    ss << *pinfo << "\n";
    ss << *gs << "\n";
    fprintf(out, "%s", ss.str().c_str());
  }

  
  int __PSGridGetID(__PSGridMPI *g) {
    return ((GridMPI *)g)->id();
  }

  __PSGridMPI *__PSGetGridByID(int id) {
    return (__PSGridMPI*)gs->FindGrid(id);
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

  PSIndex PSGridDim(void *p, int d) {
    Grid *g = (Grid *)p;    
    return g->size_[d];
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

  void __PSLoadSubgrid(__PSGridMPI *g, const __PSGridRange *gr,
                       int reuse) {
    // NOTE: This should be very rare. Not sure it should actually be
    // supported either.
    PSAssert(0 && "Not implemented yet");
    return;
  }


#ifdef __cplusplus
}
#endif

