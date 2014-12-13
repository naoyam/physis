// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_PHYSIS_COMMON_H_
#define PHYSIS_PHYSIS_COMMON_H_

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "physis/config.h"
#include "physis/stopwatch.h"
#include "physis/types.h"
#include "physis/reduce.h"
// runtime.h functions are not used in the original user code, but it
// is included to make code generation easier
#include "physis/runtime.h"

/*
 * Declarations that are common both in user input code and generated
 * code. 
 */


// TODO: Most of the code below is not necessary for the
// translator. Reorganization should be applied.

#define PS_MAX_DIM (3)

//#define PHYSIS_INDEX_INT64
// Index type is 32-bit int by default
#if ! defined(PHYSIS_INDEX_INT64)
#define PHYSIS_INDEX_INT32
#endif

#define PSAssert(e) assert(e)

#ifdef __cplusplus
extern "C" {
#endif

#if defined(PHYSIS_INDEX_INT32)
  //typedef int32_t index_t;
  typedef int32_t PSIndex;
#define PSINDEX_MAX INT32_MAX
#define PSINDEX_MIN INT32_MIN
#elif defined(PHYSIS_INDEX_INT64)
  //typedef int64_t index_t;
  typedef int64_t PSIndex;
#define PSINDEX_MAX INT64_MAX
#define PSINDEX_MIN INT64_MIN
#endif
  
  enum physis_error_code {
    PHYSIS_OUTOFMEMORY= 1
  };

  typedef int PSVectorInt[PS_MAX_DIM];
  //typedef PSIndex PSIndexVector[PS_MAX_DIM];  
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
    PSIndex min[PS_MAX_DIM];
    PSIndex max[PS_MAX_DIM];
    PSIndex local_min[PS_MAX_DIM];
    PSIndex local_max[PS_MAX_DIM];
  } __PSDomain;
  typedef __PSDomain PSDomain1D;
  typedef __PSDomain PSDomain2D;
  typedef __PSDomain PSDomain3D;

  extern PSDomain1D PSDomain1DNew(PSIndex minx, PSIndex maxx);
  extern PSDomain2D PSDomain2DNew(PSIndex minx, PSIndex maxx,
                                  PSIndex miny, PSIndex maxy);
  extern PSDomain3D PSDomain3DNew(PSIndex minx, PSIndex maxx,
                                  PSIndex miny, PSIndex maxy,
                                  PSIndex minz, PSIndex maxz);

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
      PSIndex diff = d->local_max[dividing_dim] - d->local_min[dividing_dim];
      PSIndex chunk = diff / factor;
      PSIndex rem = diff % factor;
      bd.local_min[dividing_dim] = bd.local_min[dividing_dim] + chunk * offset
	+ ((offset < rem)? offset : rem);
      bd.local_max[dividing_dim] =
	bd.local_min[dividing_dim] + chunk + ((offset < rem)? 1 : 0);
    }
    return bd;
  }

  typedef struct {
    int num;
    PSIndex offsets[(PS_MAX_DIM * 2 + 1) * PS_MAX_DIM * 2];
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
    // just a dummy constant to avoid compile errors on empty enum
    // declarations
    PS_GRID_ATTRIBUTE_DUMMY = 1 << 0 
  };
  
#define INVALID_GRID (NULL)

#ifdef __cplusplus
}
#endif

#endif /* PHYSIS_PHYSIS_COMMON_H */
