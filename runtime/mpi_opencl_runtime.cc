#include "runtime/runtime_common.h"
#include "physis/physis_mpi_opencl.h"
#include "runtime/mpi_opencl_runtime.h"

using std::vector;
using std::map;
using std::string;

using physis::IntArray;
using physis::IndexArray;
using namespace physis::runtime;

namespace physis {
namespace runtime {

ProcInfo *pinfo;
MasterMPIOpenCL *master;
ClientMPIOpenCL *client;
GridSpaceMPIOpenCL *gs;

__PSStencilRunClientFunction *__PS_stencils;
} // namespace runtime
} // namespace physis


namespace physis {
namespace runtime {

CLMPIbaseinfo *clinfo_generic;
CLMPIbaseinfo *clinfo_inner;
CLMPIbaseinfo *clinfo_boundary_copy;
std::vector<CLMPIbaseinfo *> clinfo_boundary_kernel;

CLMPIbaseinfo *clinfo_nowusing;

} // namespace runtime
} // namespace physis

namespace {

int GetNumberOfLocalProcesses(int *argc, char ***argv) {
  vector<string> opts;
  string option_string = "physis-nlp";
  int nlp = 1; // default
  if (ParseOption(argc, argv, option_string,
                  1, opts)) {
    LOG_VERBOSE() << option_string << ": " << opts[1] << "\n";
    nlp = physis::toInteger(opts[1]);
  }
  LOG_DEBUG() << "Number of local processes: "  << nlp << "\n";
  return nlp;
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

} // namespace

#ifdef __cplusplus
extern "C" {
#endif
  unsigned int num_clinfo_boundary_kernel;

  // Assumes extra arguments. The first argument is the number of
  // dimensions, and each of the remaining ones is the size of
  // respective dimension.
  //
  // For example:
  // PSInit(&argc, &argv, <dimension>, 64, 48, 40, <num_of_clients>, stencil_clients);
  //
  void PSInit(int *argc, char ***argv, int grid_num_dims, ...) {
    int rank;
    int num_procs;

    // by default the number of dimensions of processes and grids is
    // the same 
    int proc_num_dims;
    va_list vl;
    //const int *grid_size;
    IntArray grid_size;
    
    // Common treatment
    physis::runtime::PSInitCommon(argc, argv);
            
    va_start(vl, grid_num_dims);
    // Grid size
    for (int i = 0; i < grid_num_dims; ++i) {
      grid_size[i] = va_arg(vl, int);
    }
    // iteration
    int num_stencil_run_calls = va_arg(vl, int);
    // Stencil functions
    __PSStencilRunClientFunction *stencil_funcs
        = va_arg(vl, __PSStencilRunClientFunction*);
    va_end(vl);
    
    // MPI initialization
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Register num_procs, rank
    pinfo = new ProcInfo(rank, num_procs);
    LOG_INFO() << *pinfo << "\n";

    // Usually proc_size should be {1, 1, num_procs}
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

    // Default: 1
    int nlp = GetNumberOfLocalProcesses(argc, argv);

    // Initialize OpenCL
    InitOpenCL(rank, nlp, argc, argv);

    // For example
    // grid_num_dims: 3 <dimension>
    // grid_size: {64, 64, 64}
    // proc_num_dims: 3 <dimension>
    // proc_size: {1, 1, 6}
    gs = new GridSpaceMPIOpenCL(
        grid_num_dims, grid_size,
        proc_num_dims, proc_size, rank,
        clinfo_generic);

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
      client = new ClientMPIOpenCL(
          *pinfo, gs, MPI_COMM_WORLD,
          clinfo_inner);
      client->Listen();
      master = NULL;
    } else {
      LOG_INFO() << "Master process launched.\n";        
      master = new MasterMPIOpenCL(
          *pinfo, gs, MPI_COMM_WORLD, clinfo_generic);
      client = NULL;
    }
    
  }

  void PSFinalize() {
    master->Finalize();
    DestroyOpenCL();
  }

  PSDomain1D PSDomain1DNew(PSIndex minx, PSIndex maxx) {
    IndexArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IntArray(minx));
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(maxx));
    // No corresponding local region
    if (local_min >= local_max) {
      local_min.assign(0);
      local_max.assign(0);
    }
    PSDomain1D d = {{minx}, {maxx}, {local_min[0]}, {local_max[0]}};
    return d;
  }
  
  PSDomain2D PSDomain2DNew(PSIndex minx, PSIndex maxx,
                           PSIndex miny, PSIndex maxy) {
    IndexArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IntArray(minx, miny));    
    IndexArray local_max = gs->my_offset() + gs->my_size();
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

  PSDomain3D PSDomain3DNew(PSIndex minx, PSIndex maxx,
                           PSIndex miny, PSIndex maxy,
                           PSIndex minz, PSIndex maxz) {
    IndexArray local_min = gs->my_offset();
    local_min.SetNoLessThan(IntArray(minx, miny, minz));        
    IndexArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(maxx, maxy, maxz));    
    // No corresponding local region
    // TODO: dom may have a smaller number of dimensions than
    // the grid space, but __PSDomain does not know number of dimension.    
    if (!(local_min.LessThan(local_max, gs->num_dims()))) {
      local_min.assign(0);
      local_max.assign(0);
    }
    PSDomain3D d = {{minx, miny, minz}, {maxx, maxy, maxz},
                    {local_min[0], local_min[1], local_min[2]},
                    {local_max[0], local_max[1], local_max[2]}};
    return d;
  }

  // TODO: why this is required? The above DomainNew method sets
  // the local size too. 
  void __PSDomainSetLocalSize(__PSDomain *dom) {
    IntArray local_min = gs->my_offset();
    IntArray global_min(dom->min);
    local_min.SetNoLessThan(global_min);
    IntArray local_max = gs->my_offset() + gs->my_size();
    local_max.SetNoMoreThan(IntArray(dom->max));
    // No corresponding local region
    // TODO: dom may have a smaller number of dimensions than
    // the grid space, but __PSDomain does not know number of dimension.
    if (!(local_min.LessThan(local_max, gs->num_dims()))) {
      local_min.assign(0);
      local_max.assign(0);
    }
    local_min.Set(dom->local_min);
    local_max.Set(dom->local_max);
  }
  

  __PSGridMPI* __PSGridNewMPI(PSType type, int elm_size, int dim,
                              const PSVectorInt size,
                              int double_buffering,
                              int attr,
                              const PSVectorInt global_offset
                              ) {
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

  // same as mpi_runtime.cc
  void __PSGridSwap(void *p) {
    ((GridMPI *)p)->Swap();
  }

  // same as mpi_runtime.cc  
  int __PSGridGetID(__PSGridMPI *g) {
    return ((GridMPI *)g)->id();
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
    GridMPI *gm = (GridMPI*)g;
    int nd = gm->num_dims();
    va_list vl;
    va_start(vl, buf);
    IntArray index;
    for (int i = 0; i < nd; ++i) {
      index[i] = va_arg(vl, PSIndex);
    }
    va_end(vl);
    master->GridSet(gm, buf, index);
  }
  

  // same as mpi_runtime.cc
  PSIndex PSGridDim(void *p, int d) {
    Grid *g = (Grid *)p;    
    return g->size_[d];
  }

  // same as mpi_runtime.cc  
  void PSGridFree(void *p) {
    master->GridDelete((GridMPI*)p);
  }

  // same as mpi_runtime.cc  
  void PSGridCopyin(void *g, const void *buf) {
    master->GridCopyin((GridMPI*)g, buf);
    return;
  }

  // same as mpi_runtime.cc  
  void PSGridCopyout(void *g, void *buf) {
    master->GridCopyout((GridMPI*)g, buf);
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

  // same as mpi_runtime.cc
  void __PSRegisterStencilRunClient(int id, void *fptr) {
    __PS_stencils[id] = (__PSStencilRunClientFunction)fptr;
  }

  // same as mpi_runtime.cc  
  void __PSLoadNeighbor(__PSGridMPI *g,
                        const PSVectorInt halo_fw_width,
                        const PSVectorInt halo_bw_width,
                        int diagonal, int reuse, int overlap,
                        int periodic) {
    GridMPI *gm = (GridMPI*)g;
    CLbaseinfo *strm = 0;
    if (overlap) {
      strm = clinfo_boundary_copy;
    }
    gs->LoadNeighbor(gm, IntArray(halo_fw_width),
                     IntArray(halo_bw_width),
                     (bool)diagonal, reuse, periodic,
                     NULL, NULL, strm);
    return;
  }

  void __PSLoadNeighborStage1(__PSGridMPI *g,
                              const PSVectorInt halo_fw_width,
                              const PSVectorInt halo_bw_width,
                              int diagonal, int reuse, int overlap,
                              int periodic) {
    GridMPI *gm = (GridMPI*)g;
    CLbaseinfo *strm = 0;
    if (overlap) {
      strm = clinfo_boundary_copy;
    }
    gs->LoadNeighborStage1(gm, IntArray(halo_fw_width),
                           IntArray(halo_bw_width),
                           (bool)diagonal, reuse, periodic,
                           NULL, NULL, strm);
    return;
  }

  void __PSLoadNeighborStage2(__PSGridMPI *g,
                              const PSVectorInt halo_fw_width,
                              const PSVectorInt halo_bw_width,
                              int diagonal, int reuse, int overlap,
                              int periodic) {
    GridMPI *gm = (GridMPI*)g;
    CLbaseinfo *strm = 0;
    if (overlap) {
      strm = clinfo_boundary_copy;
    }
    gs->LoadNeighborStage2(gm, IntArray(halo_fw_width),
                           IntArray(halo_bw_width),
                           (bool)diagonal, reuse, periodic,
                           NULL, NULL, strm);
    return;
  }
  
  

  // same as mpi_runtime.cc  
  void __PSLoadSubgrid(__PSGridMPI *g, const __PSGridRange *gr,
                       int reuse) {
    // NOTE: This should be very rare. Not sure it should actually be
    // supported either.
    PSAssert(0 && "Not implemented yet");
    return;
  }

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
    LoadSubgrid((GridMPI*)g, gs, dims, IntArray(min_offset1, min_offset2),
                IntArray(max_offset1, max_offset2), reuse);
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

  void *__PSGridGetDev(void *g) {
    GridMPIOpenCL3D *gm = (GridMPIOpenCL3D*)g;
    return gm->GetDev();
  }

  PSIndex __PSGetLocalSize(int dim) {
    return gs->my_size()[dim];
  }

  PSIndex __PSGetLocalOffset(int dim) {
    return gs->my_offset()[dim];
  }

  __PSDomain __PSDomainShrink(__PSDomain *dom, int width) {
    __PSDomain shrinked_dom = *dom;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      shrinked_dom.local_min[i] += width;
      shrinked_dom.local_max[i] -= width;
    }
    return shrinked_dom;
  }

#ifdef __cplusplus
}
#endif

