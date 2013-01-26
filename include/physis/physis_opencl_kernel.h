/*
   Note:
   Currently this header file is not used because in physis OpenCL kernel code
   all structures are expanded into individual elements so no structure definitions
   are used.

*/

#ifndef PHYSIS_PHYSIS_OPENCL_KERNEL_H_
#define PHYSIS_PHYSIS_OPENCL_KERNEL_H_

#ifndef PHYSIS_OPENCL_KERNEL_MODE
#error "This header file must not be included except for OpenCL kernel file."
#endif

/* Using double in OpenCL kernel needs this */
/* Write the below in opencl_kernelinit.cc directly */
#if 0
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


#ifdef __cplusplus
#error "OpenCL kernel file mustn't use C++, please write in C."
#endif

/* 
  int64_t is defined in stdint.h, however
  in PHYSIS_OPENCL_KERNEL_MODE we don't want to include stdint.h
  as stdint.h includes stddef.h, stddef.h is gcc internal and is
  under gcc specific directory and clBuildProgram() won't search
  the directory by default.
*/
#ifndef int64_t
#ifdef __x86_64__
typedef long int int64_t;
#else
/* On 32 bit environ, don't define int64_t, see typedef of NUM_ELMS_T
   in physis_opencl.h
 */
#endif /* ifdef __x86_64__ */
#endif /* ifndef int64_t */

#if ! defined(PHYSIS_INDEX_INT32)
#define PHYSIS_INDEX_INT64
#endif

/* Copy only needed definitions from physis_common.h 
   TODO: some better methods?
*/
#define PHYSIS_PHYSIS_COMMON_H_ /* never include physis_common.h directly */

#if defined(PHYSIS_INDEX_INT32)
  typedef int32_t index_t;
#elif defined(PHYSIS_INDEX_INT64)
  typedef int64_t index_t;
#endif


#ifndef PS_MAX_DIM
#define PS_MAX_DIM (3)
#endif

typedef int PSVectorInt[PS_MAX_DIM];

typedef struct {
  index_t min[PS_MAX_DIM];
  index_t max[PS_MAX_DIM];
  index_t local_min[PS_MAX_DIM];
  index_t local_max[PS_MAX_DIM];
} __PSDomain;

#endif /* PHYSIS_PHYSIS_OPENCL_KERNEL_H_ */
