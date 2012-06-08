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
  T **halo_self_fw = (T**)g->_halo_self_fw();
  T **halo_self_bw = (T**)g->_halo_self_bw();
  T **halo_peer_fw = (T**)g->_halo_peer_fw();
  T **halo_peer_bw = (T**)g->_halo_peer_bw();
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
    for (int i = 0; i < g->num_dims(); ++i) {
      StringJoin sj_peer_fw, sj_peer_bw, sj_self_fw, sj_self_bw;
      int num_elms = g->halo_peer_fw_buf_size()[i] / sizeof(T);
      for (int j = 0; j < num_elms; ++j) {
        sj_peer_fw << halo_peer_fw[i][j];
      }
      num_elms = g->halo_self_fw_buf_size()[i] / sizeof(T);
      for (int j = 0; j < num_elms; ++j) {
        sj_self_fw << halo_self_fw[i][j];      
      }
      num_elms = g->halo_peer_bw_buf_size()[i] / sizeof(T);
      for (int j = 0; j < num_elms; ++j) {
        sj_peer_bw << halo_peer_bw[i][j];
      }
      num_elms = g->halo_self_bw_buf_size()[i] / sizeof(T);
      for (int j = 0; j < num_elms; ++j) {
        sj_self_bw << halo_self_bw[i][j]; 
      }
      ss << ", halo fw peer [" << i << "] {" << sj_peer_fw << "}";    
      ss << ", halo bw peer [" << i << "] {" << sj_peer_bw << "}";
      ss << ", halo fw self [" << i << "] {" << sj_self_fw << "}";    
      ss << ", halo bw self [" << i << "] {" << sj_self_bw << "}";
    }
  }
  ss << "\n";
  os << ss.str();;
  return os;
}

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_ */
