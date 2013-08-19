// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_GRID_SPACE_MPI_H_
#define PHYSIS_RUNTIME_GRID_SPACE_MPI_H_

#define __STDC_LIMIT_MACROS
#include "mpi.h"

#include "runtime/runtime_common.h"
#include "runtime/grid.h"

namespace physis {
namespace runtime {

struct FetchInfo {
  IntArray peer_index;
  IndexArray peer_offset;  
  IndexArray peer_size;
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

class GridMPI;

class GridSpaceMPI: public GridSpace {
 public:
  GridSpaceMPI(int num_dims, const IndexArray &global_size,
               int proc_num_dims, const IntArray &proc_size,
               int my_rank);
  
  virtual ~GridSpaceMPI();

  
  virtual GridMPI *CreateGrid(PSType type, int elm_size, int num_dims,
                              const IndexArray &size, 
                              const IndexArray &global_offset,
                              const IndexArray &stencil_offset_min,
                              const IndexArray &stencil_offset_max,
                              int attr);

  //! Exchange halo on one dimension of a grid asynchronously.
  /*!
    Completion is not guaranteed. Used from the synchronous
    version--ExchangeBoundaries. 
    
    \param grid Grid to exchange
    \param dim Halo dimension
    \param halo_fw_width Forward width
    \param halo_bw_width Backward width
    \param diagonal True if diagonal points are accessed.
    \param periodic True if periodic access is used.
    \param requests MPI request vector to poll completion
   */
  virtual void ExchangeBoundariesAsync(
      GridMPI *grid, int dim,
      unsigned halo_fw_width, unsigned halo_bw_width,
      bool diagonal, bool periodic,
      std::vector<MPI_Request> &requests) const;
  

  //! Exchange halo on one dimension of a grid.
  /*!
    \param grid Grid to exchange
    \param dim Halo dimension
    \param halo_fw_width Forward width
    \param halo_bw_width Backward width
    \param diagonal True if diagonal points are accessed.
    \param periodic True if periodic access is used.
   */
  virtual void ExchangeBoundaries(GridMPI *grid, int dim,
                                  unsigned halo_fw_width,
                                  unsigned halo_bw_width,
                                  bool diagonal,
                                  bool periodic) const;

  //! Exchange all boundaries of a grid.
  /*!
    \param grid_id Index of the grid to exchange.
    \param halo_width Halo width.
    \param diagonal True if diagonal points are accessed.
    \param periodic True if periodic access is used.
    \param reuse True if reuse is enabled.
   */
  virtual void ExchangeBoundaries(int grid_id,
                                  const Width2 &halo_width,
                                  bool diagonal,
                                  bool periodic,
                                  bool reuse=false) const;

  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IndexArray &offset_min,
                                const IndexArray &offset_max,
                                bool diagonal,
                                bool reuse,
                                bool periodic);
  

  virtual int FindOwnerProcess(GridMPI *g, const IndexArray &index);
  
  virtual std::ostream &Print(std::ostream &os) const;

  int num_dims() const { return num_dims_; }
  const IndexArray &global_size() const { return global_size_; }
  const IntArray &proc_size() const { return proc_size_; }
  int num_procs() const { return num_procs_; }
  const IntArray &my_idx() const { return my_idx_; }
  PSIndex **partitions() const { return partitions_; }
  PSIndex **offsets() const { return offsets_; }
  int my_rank() const { return my_rank_; }
  const IndexArray &my_size() { return my_size_; }
  const IndexArray &my_offset() { return my_offset_; }  
  const std::vector<IntArray> &proc_indices() const { return proc_indices_; }
  int GetProcessRank(const IntArray &proc_index) const;
  //! Reduce a grid with binary operator op.
  /*
   * \param out The destination scalar buffer.
   * \param op The binary operator to apply.   
   * \param g The grid to reduce.
   * \return The number of reduced elements.
   */
  virtual int ReduceGrid(void *out, PSReduceOp op, GridMPI *g);

  //virtual void Save() const;
  //virtual void Restore();

 protected:
  int num_dims_;
  IndexArray global_size_;
  int proc_num_dims_;
  IntArray proc_size_;
  int my_rank_;
  int num_procs_;
  IndexArray my_size_;  
  IndexArray my_offset_;
  PSIndex **partitions_;  
  PSIndex **offsets_;
  IndexArray min_partition_;
  IntArray my_idx_; //! Process index
  IntArray fw_neighbors_;
  IntArray bw_neighbors_;
  //! Indices for all processes; proc_indices_[my_rank] == my_idx_
  std::vector<IntArray> proc_indices_;
  MPI_Comm comm_;
  virtual void CollectPerProcSubgridInfo(
      const GridMPI *g,
      const IndexArray &grid_offset,
      const IndexArray &grid_size,
      std::vector<FetchInfo> &finfo_holder) const;
  virtual bool SendFetchRequest(FetchInfo &finfo) const;
  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);
  
  void *buf;
  size_t cur_buf_size;

  //! Calculate paritioning of a grid into sub grids.
  virtual void PartitionGrid(int num_dims, const IndexArray &size,
                             const IndexArray &global_offset,
                             IndexArray &local_offset, IndexArray &local_size);
  
};

} // namespace runtime
} // namespace physis


inline std::ostream &operator<<(std::ostream &os,
                                const physis::runtime::GridSpaceMPI &gs) {
  return gs.Print(os);
}

#endif /* PHYSIS_RUNTIME_GRID_SPACE_MPI_H_ */
