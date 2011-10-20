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
  GridMPICUDA3D(int elm_size, int num_dims, const IntArray &size,
                bool double_buffering, const IntArray &global_offset,
                const IntArray &local_offset, const IntArray &local_size,
                int attr);
  __PSGrid3DDev dev_;
  // the source is address of ordinary memory region
  virtual void Copyin(void *dst, const void *src, size_t size);
  // the dstination is address of ordinary memory region  
  virtual void Copyout(void *dst, const void *src, size_t size); 
 public:
  static GridMPICUDA3D *Create(
      int elm_size, int num_dims,  const IntArray &size,
      bool double_buffering, const IntArray &global_offset,
      const IntArray &local_offset, const IntArray &local_size,
      int attr);
  virtual ~GridMPICUDA3D();
  virtual void InitBuffer();
  virtual std::ostream &Print(std::ostream &os) const;
  __PSGrid3DDev *GetDev() { return &dev_; }
  virtual void CopyoutHalo(int dim, unsigned width, int fw, bool diagonal);
  virtual void CopyoutHalo3D0(unsigned width, int fw);
  virtual void CopyoutHalo3D1(unsigned width, int fw);
  virtual void *GetAddress(const IntArray &indices);  
  virtual void EnsureRemoteGrid(const IntArray &loal_offset,
                                const IntArray &local_size);

  void SetCUDAStream(cudaStream_t strm);
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
  GridSpaceMPICUDA(int num_dims, const IntArray &global_size,
                   int proc_num_dims, const IntArray &proc_size,
                   int my_rank);
  virtual ~GridSpaceMPICUDA();

  virtual GridMPICUDA3D *CreateGrid(int elm_size, int num_dims,
                                    const IntArray &size,
                                    bool double_buffering,
                                    const IntArray &global_offset,
                                    int attr);
  virtual bool SendBoundaries(GridMPICUDA3D *grid, int dim, unsigned width,
                              bool forward, bool diagonal,
                              ssize_t halo_size,
                              performance::DataCopyProfile &prof,
                              MPI_Request &req) const;
  
  virtual void ExchangeBoundaries(GridMPI *grid,
                                  int dim,
                                  unsigned halo_fw_width,
                                  unsigned halo_bw_width,
                                  bool diagonal) const;
  virtual void ExchangeBoundariesStage1(GridMPI *grid,
                                        int dim,
                                        unsigned halo_fw_width,
                                        unsigned halo_bw_width,
                                        bool diagonal) const;
  virtual void ExchangeBoundariesStage2(GridMPI *grid,
                                        int dim,
                                        unsigned halo_fw_width,
                                        unsigned halo_bw_width,
                                        bool diagonal) const;

  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IntArray &halo_fw_width,
                                const IntArray &halo_bw_width,
                                bool diagonal,
                                bool reuse=false,
                                const bool *fw_enabled=NULL,
                                const bool *bw_enabled=NULL,
                                cudaStream_t cuda_stream=0);
  virtual GridMPI *LoadNeighborStage1(GridMPI *g,
                                const IntArray &halo_fw_width,
                                const IntArray &halo_bw_width,
                                bool diagonal,
                                bool reuse=false,
                                const bool *fw_enabled=NULL,
                                const bool *bw_enabled=NULL,
                                cudaStream_t cuda_stream=0);
  virtual GridMPI *LoadNeighborStage2(GridMPI *g,
                                const IntArray &halo_fw_width,
                                const IntArray &halo_bw_width,
                                bool diagonal,
                                bool reuse=false,
                                const bool *fw_enabled=NULL,
                                const bool *bw_enabled=NULL,
                                cudaStream_t cuda_stream=0);
  

  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);
  virtual std::ostream& PrintLoadNeighborProf(std::ostream &os) const;
 protected:
  BufferHost *buf;
  std::map<int, performance::DataCopyProfile*> load_neighbor_prof_;
};
  
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_CUDA_H_ */
