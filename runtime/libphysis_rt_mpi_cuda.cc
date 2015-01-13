// Licensed under the BSD license. See LICENSE.txt for more details.

//#include "runtime/mpi_cuda_runtime.h"
#include "runtime/runtime_common.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/runtime_mpi_cuda.h"
#include "runtime/rpc_cuda.h"
#include "physis/physis_mpi_cuda.h"
#include "runtime/grid_mpi_cuda_exp.h"
#include "runtime/grid_space_mpi.h"

#include <cuda_runtime.h>

using std::vector;
using std::map;
using std::string;

using physis::IntArray;
using physis::IndexArray;
using namespace physis::runtime;

typedef GridMPICUDAExp GridType;
typedef GridSpaceMPICUDA<GridType> GridSpaceType;
typedef MasterCUDA<GridSpaceType> MasterType;
typedef ClientCUDA<GridSpaceType> ClientType;

#if 0
namespace physis {
namespace runtime {

ProcInfo *pinfo;
MasterMPICUDA *master;
ClientMPICUDA *client;
GridSpaceMPICUDA *gs;

__PSStencilRunClientFunction *__PS_stencils;
} // namespace runtime
} // namespace physis

#endif

namespace {

MasterType *master;
GridSpaceType *gs;

#ifdef CHECKPOINT_ENABLED
int num_local_processes;
//! Preliminary checkpoint support
/*!
  Not well tested.
  CUDA context may not be completely deleted just with
  cudaDeviceReset. 
*/
void Checkpoint() {
  gs->Save();
  CUDA_SAFE_CALL(cudaDeviceReset());
}

//! Preliminary restart support
void Restart() {
  InitCUDA(pinfo->rank(), num_local_processes);
  gs->Restore();
}
#endif

template <class T>
T __PSGridGet(__PSGridMPI *g, va_list args) {
  GridType *gm = (GridType*)g;
  int nd = gm->num_dims();
  IndexArray index;
  for (int i = 0; i < nd; ++i) {
    index[i] = va_arg(args, PSIndex);
  }
  T v;
  master->GridGet(gm, &v, index);
  return v;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

  cudaStream_t stream_inner;
  cudaStream_t stream_boundary_copy;
  int num_stream_boundary_kernel = 16;  
  cudaStream_t stream_boundary_kernel[16];

  // Assumes extra arguments. The first argument is the number of
  // dimensions, and each of the remaining ones is the size of
  // respective dimension.
  void PSInit(int *argc, char ***argv, int domain_num_dims, ...) {
#if 1    
    RuntimeMPICUDA<GridSpaceType> *rt = new RuntimeMPICUDA<GridSpaceType>();
    va_list vl;
    va_start(vl, domain_num_dims);
    rt->Init(argc, argv, domain_num_dims, vl);
    va_end(vl);
    gs = rt->gs();
    if (rt->IsMaster()) {
      master = static_cast<MasterType*>(rt->proc());
    } else {
      master = NULL;
      rt->Listen();
    }
#else     
    int rank;
    int num_procs;
    // by default the number of dimensions of processes and grids is
    // the same 
    int proc_num_dims;
    va_list vl;
    //const int *grid_size;
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

    num_local_processes = GetNumberOfLocalProcesses(argc, argv);
    
    InitCUDA(rank, num_local_processes);

    gs = new GridSpaceMPICUDA(grid_num_dims, grid_size,
                              proc_num_dims, proc_size, rank);

    LOG_INFO() << "Grid space: " << *gs << "\n";

    // Set the stencil client functions
    __PS_stencils =
        (__PSStencilRunClientFunction*)malloc(
            sizeof(__PSStencilRunClientFunction)
            * num_stencil_run_calls);
    memcpy(__PS_stencils, stencil_funcs,
           sizeof(__PSStencilRunClientFunction) * num_stencil_run_calls);
    if (rank != 0) {
      LOG_INFO() << "Client launched.\n";
      client = new ClientMPICUDA(*pinfo, gs, MPI_COMM_WORLD);
      client->Listen();
      master = NULL;
    } else {
      LOG_INFO() << "Master process launched.\n";        
      master = new MasterMPICUDA(*pinfo, gs, MPI_COMM_WORLD);
      client = NULL;
    }
#endif    
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
    if (!(local_min.LessThan(local_max, 1))) {
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
    if (!(local_min.LessThan(local_max, 2))) {
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
    if (!(local_min.LessThan(local_max, 3))) {
      local_min.Set(0);
      local_max.Set(0);
    }
    PSDomain3D d = {{minx, miny, minz}, {maxx, maxy, maxz},
                    {local_min[0], local_min[1], local_min[2]},
                    {local_max[0], local_max[1], local_max[2]}};
    return d;
  }

  //! Set the local domain size for child processes.
  /*!
    See also libphysis_rt_mpi.cc
  */
  void __PSDomainSetLocalSize(__PSDomain *dom) {
    IndexArray local_min = gs->my_offset();
    IndexArray global_min(dom->min);
    local_min.SetNoLessThan(global_min);
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IndexArray(dom->max));
    // No corresponding local region
    // TODO (Mixed dimension): dom may have a smaller number of
    // dimensions than the grid space, but __PSDomain does not know
    // number of dimension. 
    if (!(local_min.LessThan(local_max, gs->num_dims()))) {
      local_min.Set(0);
      local_max.Set(0);
    }
    local_min.CopyTo(dom->local_min);
    local_max.CopyTo(dom->local_max);
  }
  

  __PSGridMPI* __PSGridNewMPI(
      __PSGridTypeInfo *type_info,
      int dim, const PSVectorInt size, int attr,
      const PSVectorInt global_offset, const PSVectorInt stencil_offset_min,
      const PSVectorInt stencil_offset_max,
      const int *stencil_offset_min_member,
      const int *stencil_offset_max_member) {
    // Same as libphysis_rt_mpi.cc
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
    return master->GridNew(
        type_info, dim, gsize,
        IndexArray(), stencil_offset_min, stencil_offset_max,
        stencil_offset_min_member, stencil_offset_max_member,
        attr);
  }

#if 0  
  // same as mpi_runtime.cc
  void __PSGridSwap(void *p) {
    ((GridType *)p)->Swap();
  }
#endif  

  // same as mpi_runtime.cc  
  int __PSGridGetID(__PSGridMPI *g) {
    return ((GridType *)g)->id();
  }

  // same as mpi_runtime.cc  
  __PSGridMPI *__PSGetGridByID(int id) {
    return (__PSGridMPI*)gs->FindGrid(id);
  }

  // same as mpi_runtime.cc  
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

  // same as mpi_runtime.cc  
  void __PSGridSet(__PSGridMPI *g, void *buf, ...) {
    GridType *gm = (GridType*)g;
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
  

  // same as mpi_runtime.cc
  PSIndex PSGridDim(void *p, int d) {
    Grid *g = (Grid *)p;    
    return g->size()[d];
  }

  // same as mpi_runtime.cc  
  void PSGridFree(void *p) {
    master->GridDelete((GridType*)p);
  }

  // same as mpi_runtime.cc  
  void PSGridCopyin(void *g, const void *buf) {
    master->GridCopyin((GridType*)g, buf);
    return;
  }

  // same as mpi_runtime.cc  
  void PSGridCopyout(void *g, void *buf) {
    master->GridCopyout((GridType*)g, buf);;
    return;
  }

  // same as mpi_runtime.cc
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
    // testing checkpointing
    return;
  }

#if 0 // When used?  
  // same as mpi_runtime.cc
  void __PSRegisterStencilRunClient(int id, void *fptr) {
    __PS_stencils[id] = (__PSStencilRunClientFunction)fptr;
  }
#endif  

  // same as mpi_runtime.cc  
  void __PSLoadNeighbor(__PSGridMPI *g,
                        const PSVectorInt offset_min,
                        const PSVectorInt offset_max,
                        int diagonal, int reuse, int overlap,
                        int periodic) {
    GridType *gm = (GridType*)g;
    cudaStream_t strm = 0;
    if (overlap) {
      strm = stream_boundary_copy;
    }
    gs->LoadNeighbor(gm, IndexArray(offset_min),
                     IndexArray(offset_max),
                     diagonal, reuse, periodic,
                     strm);
    return;
  }

  // same as mpi_runtime.cc  
  void __PSLoadNeighborMember(__PSGridMPI *g,
                              int member,
                              const PSVectorInt offset_min,
                              const PSVectorInt offset_max,
                              int diagonal, int reuse, int overlap,
                              int periodic) {
    GridType *gm = (GridType*)g;
    cudaStream_t strm = 0;
    if (overlap) {
      strm = stream_boundary_copy;
    }
    gs->LoadNeighbor(gm, member, IndexArray(offset_min),
                     IndexArray(offset_max),
                     diagonal, reuse, periodic,
                     strm);
    return;
  }
  
#ifdef NEIGHBOR_EXCHANGE_MULTI_STAGE  
  void __PSLoadNeighborStage1(__PSGridMPI *g,
                              const PSVectorInt offset_min,
                              const PSVectorInt offset_max,
                              int diagonal, int reuse,
                              int overlap, int periodic) {
    GridType *gm = (GridType*)g;
    cudaStream_t strm = 0;
    if (overlap) {
      strm = stream_boundary_copy;
    }
    gs->LoadNeighborStage1(gm, IndexArray(offset_min),
                           IndexArray(offset_max),
                           diagonal, reuse, periodic,
                           strm);
    return;
  }

  void __PSLoadNeighborStage2(__PSGridMPI *g,
                              const PSVectorInt offset_min,
                              const PSVectorInt offset_max,
                              int diagonal, int reuse,
                              int overlap, int periodic) {
    GridType *gm = (GridType*)g;
    cudaStream_t strm = 0;
    if (overlap) {
      strm = stream_boundary_copy;
    }
    gs->LoadNeighborStage2(gm, IndexArray(offset_min),
                           IndexArray(offset_max),
                           diagonal, reuse, periodic,
                           strm);
    return;
  }
#endif  

  // same as mpi_runtime.cc  
  void __PSLoadSubgrid(__PSGridMPI *g, const __PSGridRange *gr,
                       int reuse) {
    // NOTE: This should be very rare. Not sure it should actually be
    // supported either.
    PSAssert(0 && "Not implemented yet");
    return;
  }

#if 0  
  // same as mpi_runtime.cc  
  void __PSLoadSubgrid2D(__PSGridMPI *g, 
                         int min_dim1, PSIndex min_offset1,
                         int min_dim2, PSIndex min_offset2,
                         int max_dim1, PSIndex max_offset1,
                         int max_dim2, PSIndex max_offset2,
                         int reuse) {
    PSAssert(min_dim1 == max_dim1);
    PSAssert(min_dim2 == max_dim2);
    int dims[] = {min_dim1, min_dim2};
    LoadSubgrid((GridType*)g, gs, dims, IndexArray(min_offset1, min_offset2),
                IndexArray(max_offset1, max_offset2), reuse);
    return;
  }

  // same as mpi_runtime.cc
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
    LoadSubgrid((GridType*)g, gs, dims, IndexArray(min_offset1, min_offset2, min_offset3),
                IndexArray(max_offset1, max_offset2, max_offset3), reuse);
    return;
  }
#endif  

#if 0  
  void __PSActivateRemoteGrid(__PSGridMPI *g, int active) {
    GridType *gm = (GridType*)g;
    gm->remote_grid_active() = active;
  }
#endif
  
  int __PSIsRoot() {
    return master != NULL;
  }

  void *__PSGridGetDev(void *g) {
    GridType *gm = (GridType*)g;
    return gm->GetDev();
  }
  
  PSIndex __PSGetLocalSize(int dim) {
    return gs->my_size()[dim-1];
  }

  PSIndex __PSGetLocalOffset(int dim) {
    return gs->my_offset()[dim-1];
  }

  __PSDomain __PSDomainShrink(__PSDomain *dom, int width) {
    __PSDomain shrinked_dom = *dom;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      shrinked_dom.local_min[i] += width;
      shrinked_dom.local_max[i] -= width;
    }
    return shrinked_dom;
  }

  static void __PSReduceGrid(void *buf, enum PSReduceOp op,
			     __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridType*)g);
  }

  // Have different functions for different grid types since the REF/CUDA runtimes do. 
  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGridMPI *g) {
    __PSReduceGrid(buf, op, g);
  }
  
  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGridMPI *g) {
    __PSReduceGrid(buf, op, g);    
  }
  
  void __PSReduceGridInt(void *buf, enum PSReduceOp op,
			 __PSGridMPI *g) {
    __PSReduceGrid(buf, op, g);    
  }

  void __PSReduceGridLong(void *buf, enum PSReduceOp op,
			  __PSGridMPI *g) {
    __PSReduceGrid(buf, op, g);    
  }
  
#ifdef __cplusplus
}
#endif

