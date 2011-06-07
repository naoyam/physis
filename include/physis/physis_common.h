#ifndef PHYSIS_PHYSIS_COMMON_H_
#define PHYSIS_PHYSIS_COMMON_H_

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "physis/config.h"
#include "physis/stopwatch.h"
// runtime.h functions are not used in the original user code, but it
// is included to make code generation easier
#include "physis/runtime.h"


// TODO: Most of the code below is not necessary for the
// translator. Reorganization should be applied.

#define PS_MAX_DIM (3)
#define PSDOMAIN_TYPE_NAME "__PSDomain"
#define PSDOMAIN1D_TYPE_NAME "PSDomain1D"
#define PSDOMAIN2D_TYPE_NAME "PSDomain2D"
#define PSDOMAIN3D_TYPE_NAME "PSDomain3D"

// Index type is 64-bit int by default
#if ! defined(PHYSIS_INDEX_INT32)
#define PHYSIS_INDEX_INT64
#endif

#define PSAssert(e) assert(e)

#ifdef __cplusplus
extern "C" {
#endif

#if defined(PHYSIS_INDEX_INT32)
  typedef int32_t index_t;
#elif defined(PHYSIS_INDEX_INT64)
  typedef int64_t index_t;
#endif
  
  enum physis_error_code {
    PHYSIS_OUTOFMEMORY= 1
  };

  typedef int PSVectorInt[PS_MAX_DIM];
  typedef index_t PSIndexVector[PS_MAX_DIM];  
  typedef PSVectorInt PSPoint;

  static inline void PSVectorIntInit(PSVectorInt vec, int x) {
    int i;
    for (i = 0; i < PS_MAX_DIM; i++) {
      vec[i] = x;
    }
  }
  static inline void PSVectorIntCopy(PSVectorInt dst, const PSVectorInt src) {
    memcpy(dst, src, sizeof(PSVectorInt));
  }
  
  static inline void PSPointInit(PSPoint p, int x) {
    PSVectorIntInit(p, x);
  }

  extern void PSInit(int *argc, char ***argv, int grid_num_dims, ...);
  extern void PSFinalize();

  extern void PSPrintInternalInfo(FILE *out);

  extern void PSGridCopyin(void *g, const void *src_array);
  extern void PSGridCopyout(void *g, void *dst_array);
  //extern int PSGridDim(void *g, int d);
  extern void PSGridFree(void *p);  

  typedef struct {
    index_t min[PS_MAX_DIM];
    index_t max[PS_MAX_DIM];
    index_t local_min[PS_MAX_DIM];
    index_t local_max[PS_MAX_DIM];
  } __PSDomain;
  typedef __PSDomain PSDomain1D;
  typedef __PSDomain PSDomain2D;
  typedef __PSDomain PSDomain3D;

  extern PSDomain1D PSDomain1DNew(index_t minx, index_t maxx);
  extern PSDomain2D PSDomain2DNew(index_t minx, index_t maxx,
                                  index_t miny, index_t maxy);
  extern PSDomain3D PSDomain3DNew(index_t minx, index_t maxx,
                                  index_t miny, index_t maxy,
                                  index_t minz, index_t maxz);

  static inline __PSDomain __PSDomainGetBoundary(
      __PSDomain *d, int dim, int right, int width, 
      int factor, int offset) {
    __PSDomain bd = *d;
    if (bd.local_min[dim] == 0 &&
	bd.local_max[dim] == 0) {
      // no compute part for this process
      return bd;
    }
    if (right) {
      bd.local_min[dim] = bd.local_max[dim] - width;
      PSAssert(bd.local_min[dim] >= 0);
    } else {
      bd.local_max[dim] = bd.local_min[dim] + width;
    }
    if (factor > 1) {
      int dividing_dim = 2;
      index_t diff = d->local_max[dividing_dim] - d->local_min[dividing_dim];
      index_t chunk = diff / factor;
      index_t rem = diff % factor;
      bd.local_min[dividing_dim] = bd.local_min[dividing_dim] + chunk * offset
	+ ((offset < rem)? offset : rem);
      bd.local_max[dividing_dim] =
	bd.local_min[dividing_dim] + chunk + ((offset < rem)? 1 : 0);
    }
    return bd;
  }

  typedef struct {
    int num;
    index_t offsets[(PS_MAX_DIM * 2 + 1) * PS_MAX_DIM * 2];
  } __PSOffsets;

  typedef struct {
    int num_dims;
    __PSOffsets min_offsets[PS_MAX_DIM];
    __PSOffsets max_offsets[PS_MAX_DIM];
  } __PSGridRange;

  static inline void PSAbort(int code) {
    exit(code);
  }

  enum PS_GRID_ATTRIBUTE {
    PS_GRID_PERIODIC = 1 << 0
  };
  
#define INVALID_GRID (NULL)

#ifdef __cplusplus
}
#endif

#endif /* PHYSIS_PHYSIS_COMMON_H */
