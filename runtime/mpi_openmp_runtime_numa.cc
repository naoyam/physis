#if ! defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "runtime/mpi_openmp_runtime.h"

#ifdef USE_OPENMP_NUMA
#if 0
#include <numa.h>
#endif
#endif

#include <omp.h>
#include <sched.h>

#ifdef __cplusplus
extern "C" {
#endif

  void __PSInitLoop_OpenMP(void){
#ifndef USE_OPENMP_NUMA
    return;
#else
#if 1
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int thread_idx = omp_get_thread_num();
    CPU_SET(thread_idx, &mask);
    //LOG_DEBUG() << "Resetting using CPU ID to " << thread_idx << "\n";
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1){
      perror("sched_setaffinity ");
      LOG_DEBUG() << "Calling sched_setaffinity failed for OpenMP thread "
                  << thread_idx << "\n";
    }
#endif
#endif /* ifndef USE_OPENMP_NUMA */
  } // __PSInitLoop_NUMA

#ifdef __cplusplus
}
#endif

