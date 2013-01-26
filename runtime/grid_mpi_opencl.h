// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_GRID_MPI_OpenCL_H_
#define PHYSIS_RUNTIME_GRID_MPI_OpenCL_H_

#include "runtime/grid_mpi.h"
#include "runtime/buffer.h"
#include "runtime/buffer_opencl.h"
#include "runtime/timing.h"
#include "physis/physis_mpi_opencl.h"

#if 0
#define USE_MAPPED 1
#endif

namespace physis {
namespace runtime {

class GridSpaceMPIOpenCL;

class GridMPIOpenCL3D: public GridMPI {
  friend class GridSpaceMPIOpenCL;
 protected:
  GridMPIOpenCL3D(
      PSType type, int elm_size, int num_dims, const IntArray &size,
      bool double_buffering, const IntArray &global_offset,
      const IntArray &local_offset, const IntArray &local_size,
      int attr,
      CLbaseinfo *clinfo_in);
  __PSGrid3DDev dev_;
  // the source is address of ordinary memory region
  virtual void Copyin(void *dst, const void *src, size_t size);
  virtual void Copyin_CL(cl_mem dst_clmem, size_t offset, const void *src, size_t size);
  virtual void Copyin_CL(cl_mem dst_clmem, const void *src, size_t size);
  // the dstination is address of ordinary memory region  
  virtual void Copyout(void *dst, const void *src, size_t size);
  virtual void Copyout_CL(void *dst, size_t offset, cl_mem src_clmem, size_t size);
  virtual void Copyout_CL(void *dst, cl_mem src_clmem, size_t size);
  void *GetAddress(const IntArray &indices);
  void GetAddress_CL(
      const IntArray &indices,
      cl_mem *buf_clmem_ret,
      size_t *offset_ret);
 public:
  virtual void Set(const IntArray &indices, const void *buf);
  virtual void Get(const IntArray &indices, void *buf);
  static GridMPIOpenCL3D *Create(
      PSType type, int elm_size, int num_dims,  const IntArray &size,
      bool double_buffering, const IntArray &global_offset,
      const IntArray &local_offset, const IntArray &local_size,
      int attr,
      CLbaseinfo *clinfo_in);
  virtual ~GridMPIOpenCL3D();
  virtual void InitBuffer();
  virtual std::ostream &Print(std::ostream &os) const;
  __PSGrid3DDev *GetDev() { return &dev_; }
  // TODO: Not read in detail yet!!
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);
  virtual void CopyoutHalo3D0(unsigned width, bool fw);
  virtual void CopyoutHalo3D1(unsigned width, bool fw);
  virtual void EnsureRemoteGrid(const IntArray &loal_offset,
                                const IntArray &local_size);

  void SetOpenCLinfo(CLbaseinfo *clinfo_in);

#if 0  
  virtual int Reduce(PSReduceOp op, void *out);
  
  // REFACTORING: this is an ugly fix to make things work...
  //protected:
#endif
 protected:
  cl_mem *clmem_halo_peer_fw_; // recv buffer for forward halo accesses
  cl_mem *clmem_halo_peer_bw_; // recv buffer for backward halo accesses
  CLbaseinfo *grid_clinfo_;

 public:
  cl_mem *_clmem_halo_peer_fw() { return clmem_halo_peer_fw_; }
  cl_mem *_clmem_halo_peer_bw() { return clmem_halo_peer_bw_; }


 public:
#if 0  // USE_MAPPED
  BufferOpenCLHostMapped *(*halo_self_opencl_)[2];
#else  
  BufferOpenCLHost *(*halo_self_opencl_)[2];
#endif
  BufferHost *(*halo_self_mpi_)[2];
  BufferOpenCLHost *(*halo_peer_opencl_)[2];
  BufferOpenCLDev *(*halo_peer_dev_)[2];
  virtual void FixupBufferPointers();
};

class GridSpaceMPIOpenCL: public GridSpaceMPI {
 public:
  using GridSpaceMPI::ExchangeBoundaries;
  GridSpaceMPIOpenCL(
      int num_dims, const IntArray &global_size,
      int proc_num_dims, const IntArray &proc_size,
      int my_rank,
      CLbaseinfo *clinfo_in);
  virtual ~GridSpaceMPIOpenCL();

  virtual GridMPIOpenCL3D *CreateGrid(
      PSType type, int elm_size, int num_dims,
      const IntArray &size,
      bool double_buffering,
      const IntArray &global_offset,
      int attr,
      CLbaseinfo *clinfo);
  virtual GridMPIOpenCL3D *CreateGrid(
      PSType type, int elm_size, int num_dims,
      const IntArray &size,
      bool double_buffering,
      const IntArray &global_offset,
      int attr);
  virtual bool SendBoundaries(GridMPIOpenCL3D *grid, int dim, unsigned width,
                              bool forward, bool diagonal, bool periodic,
                              ssize_t halo_size,
                              performance::DataCopyProfile &prof,
                              MPI_Request &req) const;
  virtual bool RecvBoundaries(GridMPIOpenCL3D *grid, int dim, unsigned width,
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

  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IndexArray &halo_fw_width,
                                const IndexArray &halo_bw_width,
                                bool diagonal,
                                bool reuse,
                                bool periodic,
                                const bool *fw_enabled=NULL,
                                const bool *bw_enabled=NULL,
                                CLbaseinfo *cl_stream=0);
  virtual GridMPI *LoadNeighborStage1(GridMPI *g,
                                      const IntArray &halo_fw_width,
                                      const IntArray &halo_bw_width,
                                      bool diagonal,
                                      bool reuse,
                                      bool periodic,
                                      const bool *fw_enabled=NULL,
                                      const bool *bw_enabled=NULL,
                                      CLbaseinfo *cl_stream = 0);
  virtual GridMPI *LoadNeighborStage2(GridMPI *g,
                                      const IntArray &halo_fw_width,
                                      const IntArray &halo_bw_width,
                                      bool diagonal,
                                      bool reuse,
                                      bool periodic,
                                      const bool *fw_enabled=NULL,
                                      const bool *bw_enabled=NULL,
                                      CLbaseinfo *cl_stream=0);
  

  virtual void HandleFetchRequest(GridRequest &req, GridMPI *g);
  virtual void HandleFetchReply(GridRequest &req, GridMPI *g,
                                std::map<int, FetchInfo> &fetch_map,  GridMPI *sg);
  virtual std::ostream& PrintLoadNeighborProf(std::ostream &os) const;

#if 0
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
  BufferHost *buf;
  CLbaseinfo *space_clinfo_;
  std::map<int, performance::DataCopyProfile*> load_neighbor_prof_;
};
  
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_OpenCL_H_ */
