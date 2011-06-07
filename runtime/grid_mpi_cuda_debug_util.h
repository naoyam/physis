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

#include "runtime/grid_mpi_cuda.h"
#include "physis/physis_util.h"

namespace physis {
namespace runtime {

template <class T> inline
std::ostream& print_grid(GridMPICUDA3D *g, int my_rank, std::ostream &os) {
  int nd = g->num_dims();
  BufferHost host(nd, g->elm_size());
  static_cast<BufferCUDADev3D*>(g->buffer())
      ->Copyout(host, IntArray(), g->local_size());
  T *data = (T*)host.Get();
  T **halo_peer_fw = new T*[nd];
  T **halo_peer_bw = new T*[nd];  
  for (int i = 0; i < nd; ++i) {
    BufferHost *peer_fw_buf = new BufferHost(1, g->elm_size());
    g->halo_peer_cuda_[i][1]->Copyout(
        *peer_fw_buf, IntArray(), g->halo_peer_cuda_[i][1]->size());
    halo_peer_fw[i] = (T*)peer_fw_buf->Get();
    BufferHost *peer_bw_buf = new BufferHost(1, g->elm_size());
    g->halo_peer_cuda_[i][0]->Copyout(
        *peer_bw_buf, IntArray(), g->halo_peer_cuda_[i][0]->size());
    halo_peer_bw[i] = (T*)peer_bw_buf->Get();
  }

  IntArray lsize = g->local_size();
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
      int num_elms = g->halo_peer_cuda_[i][1]->size()[0];
      for (int j = 0; j < num_elms; ++j) {
        sj_peer_fw << halo_peer_fw[i][j];
      }
      ss << ", halo fw peer [" << i << "] {" << sj_peer_fw << "}";          
      num_elms = g->halo_peer_cuda_[i][0]->size()[0];      
      for (int j = 0; j < num_elms; ++j) {
        sj_peer_bw << halo_peer_bw[i][j];
      }
      ss << ", halo bw peer [" << i << "] {" << sj_peer_bw << "}";
    }

  }
  ss << "\n";
  os << ss.str();;
  return os;
}
} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_GRID_MPI_DEBUG_UTIL_H_ */
