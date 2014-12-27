// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/runtime_common.h"
#include "physis/physis_ref.h"
#include "runtime/reduce.h"
#include "runtime/runtime_ref.h"
#include "runtime/grid.h"

#include <stdarg.h>
#include <functional>
#include <boost/function.hpp>

using namespace physis::runtime;

namespace {

RuntimeRef<GridSpace> *rt;

template <class T>
void PSReduceGridTemplate(void *buf, PSReduceOp op,
                          __PSGrid *g) {
  boost::function<T (T, T)> func = GetReducer<T>(op);
  T *d = (T *)g->p;
  T v = d[0];
  for (int64_t i = 1; i < g->num_elms; ++i) {
    v = func(v, d[i]);
  }
  *((T*)buf) = v;
  return;
}

}

#ifdef __cplusplus
extern "C" {
#endif

  void PSInit(int *argc, char ***argv, int grid_num_dims, ...) {
    rt = new RuntimeRef<GridSpace>();
    va_list vl;
    va_start(vl, grid_num_dims);
    rt->Init(argc, argv, grid_num_dims, vl);
  }
  void PSFinalize() {
    delete rt;
  }

  // Id is not used on shared memory 
  int __PSGridGetID(__PSGrid *g) {
    return 0;
  }

  __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim) {
    __PSGrid *g = (__PSGrid*)malloc(sizeof(__PSGrid));
    g->elm_size = elm_size;    
    g->num_dims = num_dims;
    PSVectorIntCopy(g->dim, dim);
    g->num_elms = 1;
    int i;
    for (i = 0; i < num_dims; i++) {
      g->num_elms *= dim[i];
    }

    g->p = calloc(g->num_elms, g->elm_size);
    if (!g->p) {
      return INVALID_GRID;
    }

    return g;
  }

  void PSGridFree(void *p) {
    __PSGrid *g = (__PSGrid *)p;        
    if (g->p) {
      free(g->p);
    }
    g->p = NULL;
  }

  void PSGridCopyin(void *p, const void *src_array) {
    __PSGrid *g = (__PSGrid *)p;
    memcpy(g->p, src_array, g->elm_size * g->num_elms);
  }

  void PSGridCopyout(void *p, void *dst_array) {
    __PSGrid *g = (__PSGrid *)p;
    memcpy(dst_array, g->p, g->elm_size * g->num_elms);
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
    memcpy(((char *)g->p) + offset, buf, g->elm_size);
  }

  
  void __PSReduceGridFloat(void *buf, PSReduceOp op,
                           __PSGrid *g) {
    PSReduceGridTemplate<float>(buf, op, g);
  }

  void __PSReduceGridDouble(void *buf, PSReduceOp op,
                            __PSGrid *g) {
    PSReduceGridTemplate<double>(buf, op, g);
  }
  
  void __PSReduceGridInt(void *buf, PSReduceOp op,
                         __PSGrid *g) {
    PSReduceGridTemplate<int>(buf, op, g);
  }

  void __PSReduceGridLong(void *buf, PSReduceOp op,
                          __PSGrid *g) {
    PSReduceGridTemplate<long>(buf, op, g);
  }

#ifdef __cplusplus
}
#endif
