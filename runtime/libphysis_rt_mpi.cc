// Licensed under the BSD license. See LICENSE.txt for more details.

#include <stdarg.h>
#include <map>
#include <string>

#include "mpi.h"

#include "runtime/grid_mpi_debug_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_runtime_common.h"
#include "runtime/runtime_mpi.h"

#include "physis/physis_mpi.h"
#include "physis/physis_util.h"

using std::map;
using std::string;

using namespace physis::runtime;
using physis::IndexArray;
using physis::IntArray;
using physis::SizeArray;

typedef GridSpaceMPI<GridMPI> GridSpaceMPIType;
typedef Master<GridSpaceMPIType> MasterType;
typedef Client<GridSpaceMPIType> ClientType;

namespace physis {
namespace runtime {

MasterType *master;
GridSpaceMPIType *gs;

} // namespace runtime
} // namespace physis


template <class T>
static T *__PSGridGetAddr(void *g, const IndexArray &indices) {
  GridMPI *gm = (GridMPI*)g;
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
  
  // Assumes extra arguments. The first argument is the number of
  // dimensions, and each of the remaining ones is the size of
  // respective dimension.
  void PSInit(int *argc, char ***argv, int grid_num_dims, ...) {
    RuntimeMPI<GridSpaceMPIType> *rt = new RuntimeMPI<GridSpaceMPIType>();
    va_list vl;
    va_start(vl, grid_num_dims);
    rt->Init(argc, argv, grid_num_dims, vl);
    va_end(vl);    
    gs = rt->gs();
    if (rt->IsMaster()) {
      master = static_cast<MasterType*>(rt->proc());
    } else {
      master = NULL;
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
    This is only relevant for non-root MPI processes. The root process
    has the correct local size for this dom, which is computed when
    calling PSDomainNew. The child processes do not execute
    PSDomainNew, so its local size are not correctly initialized yet. 
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
  
  void PSPrintInternalInfo(FILE *out) {
    std::ostringstream ss;
    if (master) ss << *master << "\n";
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
  

  __PSGridMPI* __PSGridNewMPI(PSType type, int elm_size, int dim,
                              const PSVectorInt size,
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
    return master->GridNew(
        type, elm_size, dim, gsize,
        IndexArray(), stencil_offset_min, stencil_offset_max,
        attr);
  }

  void __PSGridSwap(void *p) {
    // Do nothing
    //((GridMPI *)p)->Swap();
    return;
  }

  PSIndex __PSGridGetOffset1D(__PSGridMPI *g, PSIndex i1) {
    return static_cast<GridMPI*>(g)->CalcOffset<1>(IndexArray(i1));
  }
  PSIndex __PSGridGetOffset2D(__PSGridMPI *g, PSIndex i1,
                              PSIndex i2) {
    return static_cast<GridMPI*>(g)->CalcOffset<2>(
        IndexArray(i1, i2));
  }
  PSIndex __PSGridGetOffset3D(__PSGridMPI *g, PSIndex i1,
                              PSIndex i2, PSIndex i3) {
    return static_cast<GridMPI*>(g)->CalcOffset<3>(
        IndexArray(i1, i2, i3));
  }
  PSIndex __PSGridGetOffsetPeriodic1D(__PSGridMPI *g, PSIndex i1) {
    return static_cast<GridMPI*>(g)->CalcOffsetPeriodic<1>(
        IndexArray(i1));
  }
  PSIndex __PSGridGetOffsetPeriodic2D(__PSGridMPI *g, PSIndex i1,
                                      PSIndex i2) {
    return static_cast<GridMPI*>(g)->CalcOffsetPeriodic<2>(
        IndexArray(i1, i2));
  }
  PSIndex __PSGridGetOffsetPeriodic3D(__PSGridMPI *g, PSIndex i1,
                                      PSIndex i2, PSIndex i3) {
    return static_cast<GridMPI*>(g)->CalcOffsetPeriodic<3>(
        IndexArray(i1, i2, i3));
  }
  void *__PSGridGetBaseAddr(__PSGridMPI *g) {
    return static_cast<GridMPI*>(g)->_data();
  }

  void __PSLoadNeighbor(__PSGridMPI *g,
                        const PSVectorInt offset_min,
                        const PSVectorInt offset_max,
                        int diagonal, int reuse, int overlap,
                        int periodic) {
    if (overlap) LOG_INFO() << "Overlap possible, but not implemented\n";
    GridMPI *gm = (GridMPI*)g;
    gs->LoadNeighbor(gm, IndexArray(offset_min), IndexArray(offset_max),
                     (bool)diagonal, reuse, periodic);
    return;
  }

  static void __PSReduceGrid(void *buf, enum PSReduceOp op,
				   __PSGridMPI *g) {
    master->GridReduce(buf, op, (GridMPI*)g);
  }

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

