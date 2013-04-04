// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_
#define PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_

#include <iostream>
#include <sstream>

#include "runtime/grid_mpi.h"
#include "physis/physis_util.h"

namespace physis {
namespace runtime {

template <class T> inline
std::ostream& print_grid(GridMPI *g, int my_rank, std::ostream &os) {
  T *data = (T*)g->_data();
  T **halo_self_fw = (T**)g->halo_self_fw_();
  T **halo_self_bw = (T**)g->halo_self_bw_();
  T **halo_peer_fw = (T**)g->halo_peer_fw_();
  T **halo_peer_bw = (T**)g->halo_peer_bw_();
  IndexArray lsize = g->local_size();
  std::stringstream ss;
  ss << "[rank:" << my_rank << "] ";

  if (g->empty()) {
    ss << "<EMPTY>";
  } else {
    physis::StringJoin sj;
    for (ssize_t i = 0; i < lsize.accumulate(g->num_dims()); ++i) {
      sj << data[i];
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
