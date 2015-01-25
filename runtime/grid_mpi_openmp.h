// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_MPI_OPENMP_H_
#define PHYSIS_RUNTIME_GRID_MPI_OPENMP_H_

#include "runtime/grid_mpi.h"
#include "runtime/buffer_mpi_openmp.h"

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
                const IntArray &division,
                int attr);
 public:
  static GridMPIOpenMP *Create(PSType type, int elm_size,
                               int num_dims, const IntArray &size,
                               bool double_buffering,
                               const IntArray &global_offset,
                               const IntArray &local_offset,
                               const IntArray &local_size,
                               const IntArray &division,
                               int attr);
  virtual ~GridMPIOpenMP() {};
  
  virtual void CopyoutHalo2D0(unsigned width, bool fw, char *buf);
  virtual void CopyoutHalo3D0(unsigned width, bool fw, char *buf);
  virtual void CopyoutHalo3D1(unsigned width, bool fw, char *buf);
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);

  virtual void EnsureRemoteGrid(const IntArray &loal_offset,
                                const IntArray &local_size);

  virtual void *GetAddress(const IntArray &indices);

  virtual int Reduce(PSReduceOp op, void *out);

 public:
  void Copyin(void *dst, const void *src, size_t size);
  void Copyout(void *dst, const void *src, size_t size);
  void Swap();

 protected:
  char **data_MP_[2];
  IntArray division_;

 public:
  char **_data_MP_() { return data_MP_[0]; }
  char **_data_emit_MP() { return data_MP_[1]; }
  const IntArray division() const { return division_; }

 public:
  Buffer *buffer_emit() { return data_buffer_[1]; }

 protected:
  virtual void InitBuffer();
  virtual void FixupBufferPointers();

 public:
  virtual void CopyinoutSubgrid(
      bool copyout_to_buf_p,
      size_t elm_size, int num_dims,
      BufferHostOpenMP *bufmp,
      const IntArray &grid_size,
      void *buf,
      const IntArray &subgrid_offset,
      const IntArray &subgrid_size
                                );

  virtual void CopyinoutSubgrid(
      bool copyIN_FROM_BUF_p,
      size_t elm_size, int num_dims,
      void *global_buf,
      const IntArray &global_size,
      BufferHostOpenMP *subbufmp,
      const IntArray &subgrid_offset,
      const IntArray &subgrid_size
                                );

 public:
  virtual void InitNUMA(unsigned int maxMPthread);

};

class GridSpaceMPIOpenMP: public GridSpaceMPI {
 public:
  GridSpaceMPIOpenMP(
      int num_dims, const IntArray &global_size,
      int proc_num_dims, const IntArray &proc_size,
      int my_rank);
  
  virtual ~GridSpaceMPIOpenMP();

  virtual GridMPIOpenMP *CreateGrid(
      PSType type, int elm_size, int num_dims,
      const IntArray &size, bool double_buffering,
      const IntArray &global_offset,
      const IntArray &division,
      int attr);

 protected:
  virtual void CollectPerProcSubgridInfo(const GridMPI *g,
                                         const IntArray &grid_offset,
                                         const IntArray &grid_size,
                                         std::vector<FetchInfo> &finfo_holder) const;
  virtual bool SendFetchRequest(FetchInfo &finfo) const;
  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);

};

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_OPENMP_H_ */

