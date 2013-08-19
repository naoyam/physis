// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_GRID_MPI_CUDA_H_
#define PHYSIS_RUNTIME_GRID_MPI_CUDA_H_

#include "runtime/grid_mpi.h"
#include "runtime/buffer.h"
#include "runtime/buffer_cuda.h"
#include "runtime/timing.h"
#include "physis/physis_mpi_cuda.h"

#define USE_MAPPED 1

namespace physis {
namespace runtime {

class GridSpaceMPICUDA;

class GridMPICUDA3D: public GridMPI {
  friend class GridSpaceMPICUDA;
 protected:  
  GridMPICUDA3D(PSType type, int elm_size, int num_dims, const IndexArray &size,
                bool double_buffering, const IndexArray &global_offset,
                const IndexArray &local_offset, const IndexArray &local_size,
                int attr);
  __PSGridDev3D dev_;
  // the source is address of ordinary memory region
  virtual void Copyin(void *dst, const void *src, size_t size);
  // the dstination is address of ordinary memory region  
  virtual void Copyout(void *dst, const void *src, size_t size); 
 public:
  static GridMPICUDA3D *Create(
      PSType type, int elm_size, int num_dims,  const IndexArray &size,
      bool double_buffering, const IndexArray &global_offset,
      const IndexArray &local_offset, const IndexArray &local_size,
      int attr);
  virtual ~GridMPICUDA3D();
  virtual void DeleteBuffers();
  virtual void InitBuffers();
  virtual std::ostream &Print(std::ostream &os) const;
  __PSGridDev3D *GetDev() { return &dev_; }
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);
  virtual void CopyoutHalo3D0(unsigned width, bool fw);
  virtual void CopyoutHalo3D1(unsigned width, bool fw);
  virtual void *GetAddress(const IndexArray &indices);  
  virtual void EnsureRemoteGrid(const IndexArray &loal_offset,
                                const IndexArray &local_size);

  void SetCUDAStream(cudaStream_t strm);
  
  virtual int Reduce(PSReduceOp op, void *out);

  virtual void Save();
  virtual void Restore();

  
  // REFACTORING: this is an ugly fix to make things work...
  //protected:
 public:
#if USE_MAPPED  
  BufferCUDAHostMapped *(*halo_self_cuda_)[2];
#else  
  BufferCUDAHost *(*halo_self_cuda_)[2];
#endif
  BufferHost *(*halo_self_mpi_)[2];
  BufferCUDAHost *(*halo_peer_cuda_)[2];
  BufferCUDADev *(*halo_peer_dev_)[2];
  virtual void FixupBufferPointers();
};

class GridSpaceMPICUDA: public GridSpaceMPI {
 public:
  using GridSpaceMPI::ExchangeBoundaries;  
  GridSpaceMPICUDA(int num_dims, const IndexArray &global_size,
                   int proc_num_dims, const IntArray &proc_size,
                   int my_rank);
  virtual ~GridSpaceMPICUDA();

  virtual GridMPICUDA3D *CreateGrid(PSType type, int elm_size, int num_dims,
                                    const IndexArray &size,
                                    bool double_buffering,
                                    const IndexArray &global_offset,
                                    int attr);
  virtual bool SendBoundaries(GridMPICUDA3D *grid, int dim, unsigned width,
                              bool forward, bool diagonal, bool periodic,
                              ssize_t halo_size,
                              performance::DataCopyProfile &prof,
                              MPI_Request &req) const;
  virtual bool RecvBoundaries(GridMPICUDA3D *grid, int dim, unsigned width,
                              bool forward, bool diagonal, bool periodic,
                              ssize_t halo_size,
                              performance::DataCopyProfile &prof) const;
  
  virtual void ExchangeBoundaries(GridMPI *grid,
                                  int dim,
                                  unsigned halo_fw_width,
                                  unsigned halo_bw_width,
                                  bool diagonal,
                                  bool periodic) const;
  virtual void ExchangeBoundariesStage1(GridMPI *grid,
                                        int dim,
                                        unsigned halo_fw_width,
                                        unsigned halo_bw_width,
                                        bool diagonal,
                                        bool periodic) const;
  virtual void ExchangeBoundariesStage2(GridMPI *grid,
                                        int dim,
                                        unsigned halo_fw_width,
                                        unsigned halo_bw_width,
                                        bool diagonal,
                                        bool periodic) const;
  using GridSpaceMPI::LoadNeighbor;
  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IndexArray &offset_min,
                                const IndexArray &offset_max,
                                bool diagonal,
                                bool reuse,
                                bool periodic,
                                cudaStream_t cuda_stream);
  virtual GridMPI *LoadNeighborStage1(GridMPI *g,
                                      const IndexArray &offset_min,
                                      const IndexArray &offset_max,
                                      bool diagonal,
                                      bool reuse,
                                      bool periodic,
                                      cudaStream_t cuda_stream);
  virtual GridMPI *LoadNeighborStage2(GridMPI *g,
                                      const IndexArray &offset_min,
                                      const IndexArray &offset_max,
                                      bool diagonal,
                                      bool reuse,
                                      bool periodic,
                                      cudaStream_t cuda_stream);
  

  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);
  virtual std::ostream& PrintLoadNeighborProf(std::ostream &os) const;

  //! Reduce a grid with binary operator op.
  /*
   * \param out The destination scalar buffer.
   * \param op The binary operator to apply.   
   * \param g The grid to reduce.
   * \return The number of reduced elements.
   */
  virtual int ReduceGrid(void *out, PSReduceOp op, GridMPI *g);
  
 protected:
  BufferHost *buf;
  std::map<int, performance::DataCopyProfile*> load_neighbor_prof_;
};
  
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_CUDA_H_ */
