#ifndef PHYSIS_RUNTIME_GRID_MPI_OPENMP_H_
#define PHYSIS_RUNTIME_GRID_MPI_OPENMP_H_

#include "runtime/grid_mpi.h"

namespace physis {
namespace runtime {

class GridSpaceMPIOpenMP;

// NOTE: Avoid use of MPI routines if possible in this class so that
// this can be used with non-MPI parallel APIs.
class GridMPIOpenMP: public GridMPI {
  friend class GridSpaceMPIOpenMP;
 protected:
  GridMPIOpenMP(PSType type, int elm_size, int num_dims,
          const IntArray &size,
          bool double_buffering, const IntArray &global_offset,
          const IntArray &local_offset, const IntArray &local_size,
          int attr);
 public:
#if 1
  static GridMPIOpenMP *Create(PSType type, int elm_size,
                         int num_dims, const IntArray &size,
                         bool double_buffering,
                         const IntArray &global_offset,
                         const IntArray &local_offset,
                         const IntArray &local_size,
                         int attr);
#endif
  virtual ~GridMPIOpenMP() {};
  
#if 0
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
#endif
  virtual void EnsureRemoteGrid(const IntArray &loal_offset,
                                const IntArray &local_size);
#if 0
  GridMPI *remote_grid() { return remote_grid_; }
  bool &remote_grid_active() { return remote_grid_active_; }
  virtual void *GetAddress(const IntArray &indices);

  virtual void Resize(const IntArray &local_offset,
                      const IntArray &local_size);
  
  virtual int Reduce(PSReduceOp op, void *out);
#endif

 protected:
#if 0
  bool empty_;
  IntArray global_offset_;
#endif
 protected:
#if 0
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
#endif
  virtual void InitBuffer();
  virtual void FixupBufferPointers();
};

#if 0
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
#endif

class GridSpaceMPIOpenMP: public GridSpaceMPI {
 public:
  GridSpaceMPIOpenMP(int num_dims, const IntArray &global_size,
               int proc_num_dims, const IntArray &proc_size,
               int my_rank);
  
  virtual ~GridSpaceMPIOpenMP();

#if 0
  virtual void PartitionGrid(int num_dims, const IntArray &size,
                             const IntArray &global_offset,
                             IntArray &local_offset, IntArray &local_size);
  
#endif
  virtual GridMPI *CreateGrid(PSType type, int elm_size, int num_dims,
                              const IntArray &size, bool double_buffering,
                              const IntArray &global_offset,
                              int attr);

#if 0
  // These two functions perform the same thing except for requiring
  // the different type first parameters. The former may be omitted.
  virtual void ExchangeBoundariesAsync(GridMPI *grid,
                                       int dim,
                                       unsigned halo_fw_width,
                                       unsigned halo_bw_width,
                                       bool diagonal,
                                       bool periodic,
                                       std::vector<MPI_Request> &requests) const;
  
  // These two functions perform the same thing except for requiring
  // the different type first parameters. The former may be omitted.
  virtual void ExchangeBoundaries(GridMPI *grid, int dim,
                                  unsigned halo_fw_width, unsigned halo_bw_width,
                                  bool diagonal,
                                  bool periodic) const;

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
                                  bool diagonal,
                                  bool periodic,
                                  bool reuse=false) const;


  virtual GridMPI *LoadSubgrid(GridMPI *grid, const IntArray &grid_offset,
                               const IntArray &grid_size, bool reuse=false);
  
  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IntArray &halo_fw_width,
                                const IntArray &halo_bw_width,
                                bool diagonal,
                                bool reuse,
                                bool periodic,
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
  //! Reduce a grid with binary operator op.
  /*
   * \param out The destination scalar buffer.
   * \param op The binary operator to apply.   
   * \param g The grid to reduce.
   * \return The number of reduced elements.
   */
  virtual int ReduceGrid(void *out, PSReduceOp op, GridMPI *g);
#endif

 protected:
#if 0
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
#endif
  virtual void CollectPerProcSubgridInfo(const GridMPI *g,
                                         const IntArray &grid_offset,
                                         const IntArray &grid_size,
                                         std::vector<FetchInfo> &finfo_holder) const;
  virtual bool SendFetchRequest(FetchInfo &finfo) const;
  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);
#if 0
  void *buf;
  size_t cur_buf_size;
#endif
  void **buf_MULTI;
};


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_OPENMP_H_ */

