// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/buffer_mpi_openmp.h"

#ifdef USE_OPENMP_NUMA
#include <numaif.h>
#include <numa.h>
#endif

namespace physis {
namespace runtime {

void BufferHostOpenMP::MoveMultiBuffer(unsigned int maxcpunodes)
{
#ifndef USE_OPENMP_NUMA
  return;
#else
  if (mbind_done_p) return;
  mbind_done_p = true; // Never try again
  if (maxcpunodes < 2) return; // No use
  if (numa_available() == -1) return; // NUMA not available

  unsigned int total_num = 1;
  for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++)
      total_num *= division_[dim];

  int tolcpunum = sysconf(_SC_NPROCESSORS_ONLN);
  if (tolcpunum == -1) {
    LOG_DEBUG() << "Calling sysconf(_SC_NPROCESSORS_ONLN) failed,";
    LOG_DEBUG() << "using " << total_num << " as the available cpu number\n";
    tolcpunum = total_num;
  }
  else {
    LOG_DEBUG() << "sysconf reported that " << tolcpunum << " cpus available\n";
  }
  for (unsigned int cpuid = 0; cpuid < total_num; cpuid++){
    int cpuid_to_move = numa_node_of_cpu(cpuid % (unsigned int) tolcpunum);
    if (cpuid_to_move == -1) {
      LOG_DEBUG() << "Calling numa_node_of cpu failed for id " << cpuid << "\n";
      continue;
    }
    struct bitmask *numa_mask = numa_allocate_nodemask();
    numa_bitmask_setbit(numa_mask, cpuid_to_move);

    LOG_DEBUG() << "Moving allocated memory id " << cpuid
      << " to NUMA node " << cpuid_to_move << "\n";
    int errstatus = 
      mbind(
        buf_mp_[cpuid], mp_cpu_allocBytes_[cpuid],
        MPOL_BIND,
        numa_mask->maskp, numa_mask->size,
        MPOL_MF_MOVE | MPOL_MF_STRICT
      );
    if (errstatus) {
      perror("mbind :");
      LOG_DEBUG() << "Calling mbind failed for cpuid " << cpuid << " , ignoring.\n";
    }

    numa_free_nodemask(numa_mask);
  } // for (unsigned int cpuid = 0; cpuid < total_num; cpuid++)
#endif
}

} // namespace runtime
} // namespace physis
