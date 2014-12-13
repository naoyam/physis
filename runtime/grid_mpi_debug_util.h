// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_
#define PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_

#include <iostream>
#include <sstream>

#include "runtime/grid_mpi.h"
#include "physis/physis_util.h"

namespace physis {
namespace runtime {

template <class T> inline
std::ostream& PrintGrid(GridMPI *g, int my_rank, std::ostream &os) {
  IndexArray lsize = g->local_size();
  std::stringstream ss;
  ss << "[rank:" << my_rank << "] ";

  if (g->empty()) {
    ss << "<EMPTY>";
  } else {
    physis::StringJoin sj;
    if (g->num_dims() == 3) {
      for (int k = 0; k < g->local_real_size()[2]; ++k) {
        for (int j = 0; j < g->local_real_size()[1]; ++j) {
          for (int i = 0; i < g->local_real_size()[0]; ++i) {
            IndexArray ijk = IndexArray(i, j, k);
            IndexArray t = ijk + g->local_real_offset();
            sj << *((T*)(g->GetAddress(t)));
          }
        }
      }
    } else if  (g->num_dims() == 2) {
      for (int j = 0; j < g->local_size()[1]; ++j) {    
        for (int i = 0; i < g->local_size()[0]; ++i) {
          IndexArray ij = IndexArray(i, j);
          IndexArray t = ij + g->local_offset();
          sj << *((T*)(g->GetAddress(t)));
        }
      }
    } else if  (g->num_dims() == 1) {
      for (int i = 0; i < g->local_size()[0]; ++i) {
        sj << *((T*)(g->GetAddress(IndexArray(i+g->local_offset()[0]))));
      }
    } else {
      LOG_ERROR() << "Unsupported dimension\n";
      exit(1);
    }
    ss << "data {" << sj << "}";
  }
  ss << "\n";
  os << ss.str();;
  return os;
}

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_ */
