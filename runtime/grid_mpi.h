// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_GRID_MPI_H_
#define PHYSIS_RUNTIME_GRID_MPI_H_

#include "mpi.h"

#include "runtime/runtime_common.h"
#include "runtime/grid.h"

namespace physis {
namespace runtime {

class GridSpaceMPI;

// NOTE: Avoid use of MPI routines if possible in this class so that
// this can be used with non-MPI parallel APIs.
class GridMPI: public Grid {
  friend class GridSpaceMPI;
 protected:
  GridMPI(int elm_size, int num_dims, const IntArray &size,
          bool double_buffering, const IntArray &global_offset,
          const IntArray &local_offset, const IntArray &local_size,
          int attr);
 public:
  static GridMPI *Create(int elm_size, int num_dims, const IntArray &size,
                         bool double_buffering,
                         const IntArray &global_offset,
                         const IntArray &local_offset,
                         const IntArray &local_size,
                         int attr);
  virtual ~GridMPI();
  
  virtual void CopyoutHalo2D0(unsigned width, bool fw, char *buf);
  virtual void CopyoutHalo3D0(unsigned width, bool fw, char *buf);
  virtual void CopyoutHalo3D1(unsigned width, bool fw, char *buf);
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);
  size_t CalcHaloSize(int dim, int width, bool diagonal);
  virtual std::ostream &Print(std::ostream &os) const;
  const IntArray& local_size() const { return local_size_; }
  const IntArray& local_offset() const { return local_offset_; }
  bool halo_has_diagonal() const { return halo_has_diagonal_; }
  const IntArray& halo_fw_width() const { return halo_fw_width_; }
  const IntArray& halo_bw_width() const { return halo_bw_width_; }
  const IntArray& halo_fw_max_width() const { return halo_fw_max_width_; }
  const IntArray& halo_bw_max_width() const { return halo_bw_max_width_; }
  char **_halo_self_fw() { return halo_self_fw_; }
  char **_halo_self_bw() { return halo_self_bw_; }  
  char **_halo_peer_fw() { return halo_peer_fw_; }
  char **_halo_peer_bw() { return halo_peer_bw_; }
  bool empty() const { return empty_; }
  const IntArray& halo_self_fw_buf_size() const {
    return halo_self_fw_buf_size_;
  }
  const IntArray& halo_self_bw_buf_size() const {
    return halo_self_bw_buf_size_;
  }  
  const IntArray& halo_peer_fw_buf_size() const {
    return halo_peer_fw_buf_size_;
  }
  const IntArray& halo_peer_bw_buf_size() const {
    return halo_peer_bw_buf_size_;
  }
  IntArray* halo_fw_size() {
    return halo_fw_size_;
  }
  IntArray* halo_bw_size() {
    return halo_bw_size_;
  }
  void SetHaloSize(int dim, bool fw, size_t width, bool diagonal);
  virtual void EnsureRemoteGrid(const IntArray &loal_offset,
                                const IntArray &local_size);
  GridMPI *remote_grid() { return remote_grid_; }
  bool &remote_grid_active() { return remote_grid_active_; }
  virtual void *GetAddress(const IntArray &indices);

  virtual void Resize(const IntArray &local_offset,
                      const IntArray &local_size);
 protected:
  bool empty_;
  IntArray global_offset_;
  IntArray local_offset_;
  IntArray local_size_;
  bool halo_has_diagonal_;
  IntArray halo_fw_width_;
  IntArray halo_bw_width_;
  IntArray halo_fw_max_width_;
  IntArray halo_bw_max_width_;
  char **halo_self_fw_; // send buffer for forward halo accesses
  char **halo_self_bw_; // send buffer for backward halo accesses
  char **halo_peer_fw_; // recv buffer for forward halo accesses
  char **halo_peer_bw_; // recv buffer for backward halo accesses
  IntArray halo_self_fw_buf_size_;
  IntArray halo_self_bw_buf_size_;
  IntArray halo_peer_fw_buf_size_;
  IntArray halo_peer_bw_buf_size_;
  IntArray halo_fw_size_[PS_MAX_DIM];
  IntArray halo_bw_size_[PS_MAX_DIM];
  GridMPI *remote_grid_;
  // Indicates whether the remote grid is used instead of this
  // grid. This variable is set true when LoadSubgrid decides to use
  // the remote grid for a kernel, but is set false after execution of
  // the kernel. If the same subgrid is used again and the grid is
  // read-only, LoadSubgrid decides to use the remote grid, and this
  // variale is set true again. 
  bool remote_grid_active_;
  virtual void InitBuffer();
  virtual void FixupBufferPointers();
};


struct FetchInfo {
  IntArray peer_index;
  IntArray peer_offset;  
  IntArray peer_size;
};

enum GRID_REQUEST_KIND {INVALID, DONE, FETCH_REQUEST, FETCH_REPLY};

struct GridRequest {
  int my_rank;
  GRID_REQUEST_KIND kind;
  GridRequest(): my_rank(-1), kind(INVALID) {}
  GridRequest(int rank, GRID_REQUEST_KIND k): my_rank(rank), kind(k) {}
};

void SendGridRequest(int my_rank, int peer_rank, MPI_Comm comm,
                     GRID_REQUEST_KIND kind);
GridRequest RecvGridRequest(MPI_Comm comm);


class GridSpaceMPI: public GridSpace {
 public:
  GridSpaceMPI(int num_dims, const IntArray &global_size,
               int proc_num_dims, const IntArray &proc_size,
               int my_rank);
  
  virtual ~GridSpaceMPI();

  virtual void PartitionGrid(int num_dims, const IntArray &size,
                             const IntArray &global_offset,
                             IntArray &local_offset, IntArray &local_size);
  
  virtual GridMPI *CreateGrid(int elm_size, int num_dims, const IntArray &size,
                              bool double_buffering,
                              const IntArray &global_offset,
                              int attr);

  // These two functions perform the same thing except for requiring
  // the different type first parameters. The former may be omitted.
  virtual void ExchangeBoundariesAsync(GridMPI *grid,
                                       int dim,
                                       unsigned halo_fw_width,
                                       unsigned halo_bw_width,
                                       bool diagonal,
                                       std::vector<MPI_Request> &requests) const;
  
  // These two functions perform the same thing except for requiring
  // the different type first parameters. The former may be omitted.
  virtual void ExchangeBoundaries(GridMPI *grid, int dim,
                                  unsigned halo_fw_width, unsigned halo_bw_width,
                                  bool diagonal) const;

#if 0
  void ExchangeBoundariesAsync(int grid_id,
                               const IntArray &halo_fw_width,
                               const IntArray &halo_bw_width,
                               bool diagonal,
                               std::vector<MPI_Request> &requests) const;
#endif  

  virtual void ExchangeBoundaries(int grid_id,
                                  const IntArray &halo_fw_width,
                                  const IntArray &halo_bw_width,
                                  bool diagonal, bool reuse=false) const;


  virtual GridMPI *LoadSubgrid(GridMPI *grid, const IntArray &grid_offset,
                               const IntArray &grid_size, bool reuse=false);
  
  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IntArray &halo_fw_width,
                                const IntArray &halo_bw_width,
                                bool diagonal,
                                bool reuse=false,
                                const bool *fw_enabled=NULL,
                                const bool *bw_enabled=NULL);

  virtual int FindOwnerProcess(GridMPI *g, const IntArray &index);
  
  virtual std::ostream &Print(std::ostream &os) const;

  int num_dims() const { return num_dims_; }
  const IntArray &global_size() const { return global_size_; }
  const IntArray &proc_size() const { return proc_size_; }
  int num_procs() const { return num_procs_; }
  const IntArray &my_idx() const { return my_idx_; }
  index_t **partitions() const { return partitions_; }
  index_t **offsets() const { return offsets_; }
  int my_rank() const { return my_rank_; }
  const IntArray &my_size() { return my_size_; }
  const IntArray &my_offset() { return my_offset_; }  
  const std::vector<IntArray> &proc_indices() const { return proc_indices_; }
  MPI_Comm comm() const { return comm_; };
  int GetProcessRank(const IntArray &proc_index) const;
 protected:
  int num_dims_;
  IntArray global_size_;
  int proc_num_dims_;
  IntArray proc_size_;
  int my_rank_;
  int num_procs_;
  IntArray my_size_;  
  IntArray my_offset_;
  index_t **partitions_;  
  index_t **offsets_;
  IntArray min_partition_;
  IntArray my_idx_;
  IntArray fw_neighbors_;
  IntArray bw_neighbors_;
  // indices for all processes; proc_indices_[my_rank] == my_idx_
  std::vector<IntArray> proc_indices_;
  MPI_Comm comm_;
  virtual void CollectPerProcSubgridInfo(const GridMPI *g,
                                         const IntArray &grid_offset,
                                         const IntArray &grid_size,
                                         std::vector<FetchInfo> &finfo_holder) const;
  virtual bool SendFetchRequest(FetchInfo &finfo) const;
  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);
  void *buf;
  size_t cur_buf_size;
};


void LoadSubgrid(GridMPI *g, GridSpaceMPI *gs,
                 int *dims, const IntArray &min_offsets,
                 const IntArray &max_offsets,
                 bool reuse);


} // namespace runtime
} // namespace physis

inline std::ostream& operator<<(std::ostream &os,
                                physis::runtime::GridMPI &g) {
  return g.Print(os);
}


inline std::ostream &operator<<(std::ostream &os,
                                const physis::runtime::GridSpaceMPI &gs) {
  return gs.Print(os);
}

#endif /* PHYSIS_RUNTIME_GRID_MPI_H_ */

