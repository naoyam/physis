#include "runtime/grid_mpi_opencl.h"
#include "runtime/buffer.h"
#include "runtime/grid_util.h"

using namespace physis::runtime::performance;
using std::make_pair;

namespace physis {
namespace runtime {

//
// Grid
//
GridMPIOpenCL3D::GridMPIOpenCL3D(
    PSType type, int elm_size, int num_dims, const IntArray &size,
    bool double_buffering, const IntArray &global_offset,
    const IntArray &local_offset, const IntArray &local_size,
    int attr,
    CLbaseinfo *clinfo_in):
    GridMPI(type, elm_size, num_dims, size, double_buffering, global_offset,
            local_offset, local_size, attr),
    grid_clinfo_(clinfo_in)
{

  dev_.p0 = 0;  
  if (empty_) return;

  // These pointers are replaced with buffer substrate in this class
  delete[]  halo_self_fw_;
  halo_self_fw_ = NULL;
  delete[] halo_self_bw_;
  halo_self_bw_ = NULL;

  clmem_halo_peer_fw_ = new cl_mem[num_dims_];
  clmem_halo_peer_bw_ = new cl_mem[num_dims_];


#if 0 //USE_MAPPED  
  halo_self_opencl_ = new BufferOpenCLHostMapped*[num_dims_][2];
#else
  halo_self_opencl_ = new BufferOpenCLHost*[num_dims_][2];  
#endif
  halo_self_mpi_ = new BufferHost*[num_dims_][2];
  halo_peer_opencl_ = new BufferOpenCLHost*[num_dims_][2];
  halo_peer_dev_ = new BufferOpenCLDev*[num_dims_][2];
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
#if 0 //USE_MAPPED      
      halo_self_opencl_[i][j] = new BufferOpenCLHostMapped(elm_size);
#else
      halo_self_opencl_[i][j] = new BufferOpenCLHost(elm_size);      
#endif      
      halo_self_mpi_[i][j] = new BufferHost(elm_size);
      halo_peer_opencl_[i][j] = new BufferOpenCLHost(elm_size);
      halo_peer_dev_[i][j] = new BufferOpenCLDev(elm_size, clinfo_in);
    }
  }
  FixupBufferPointers();
}

GridMPIOpenCL3D *GridMPIOpenCL3D::Create(
    PSType type, int elm_size, int num_dims, const IntArray &size,
    bool double_buffering, const IntArray &global_offset,
    const IntArray &local_offset, const IntArray &local_size,
    int attr,
    CLbaseinfo *clinfo_in) {
  GridMPIOpenCL3D *gmc = new GridMPIOpenCL3D(
                                        type, elm_size, num_dims, size,
                                        double_buffering, global_offset,
                                        local_offset, local_size,
                                        attr,
                                        clinfo_in);
  gmc->InitBuffer();
  return gmc;
}


GridMPIOpenCL3D::~GridMPIOpenCL3D() {
  if (empty_) return;
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
      delete halo_self_opencl_[i][j];
      delete halo_self_mpi_[i][j];
      delete halo_peer_opencl_[i][j];
      delete halo_peer_dev_[i][j];
    }

    halo_peer_fw_[i] = NULL;
    halo_peer_bw_[i] = NULL;    
  }
  delete[] halo_self_opencl_;
  delete[] halo_self_mpi_;
  delete[] halo_peer_opencl_;
  delete[] halo_peer_dev_;

  for (int i = 0; i < num_dims_-1; ++i) {
    if (clmem_halo_peer_fw_) {
      if (clmem_halo_peer_fw_[i]) clReleaseMemObject(clmem_halo_peer_fw_[i]);
    }
    if (clmem_halo_peer_bw_) {
      if (clmem_halo_peer_bw_[i]) clReleaseMemObject(clmem_halo_peer_bw_[i]);
    }
  }
  if (clmem_halo_peer_fw_) delete[] clmem_halo_peer_fw_;
  if (clmem_halo_peer_bw_) delete[] clmem_halo_peer_bw_;
}

void GridMPIOpenCL3D::InitBuffer() {
  LOG_DEBUG() << "Initializing grid buffer\n";
  if (empty_) return;
  data_buffer_[0] = new BufferOpenCLDev3D(num_dims(), elm_size(), grid_clinfo_);
  LOG_DEBUG() << "Calling data_buffer[0]->Allocate with size " << local_size() << "\n";
  data_buffer_[0]->Allocate(local_size());
  if (double_buffering_) {
    data_buffer_[1] = new BufferOpenCLDev(num_dims(), elm_size(), grid_clinfo_);
    data_buffer_[1]->Allocate(local_size());    
  } else {
    data_buffer_[1] = data_buffer_[0];
  }
  FixupBufferPointers();
}

void GridMPIOpenCL3D::SetOpenCLinfo(CLbaseinfo *clinfo_in)
{
  static_cast<BufferOpenCLDev3D*>(data_buffer_[0])->stream_clinfo() = clinfo_in;
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
      halo_peer_dev_[i][j]->stream_clinfo() = clinfo_in;
    }
  }
}

void *GridMPIOpenCL3D::GetAddress(const IntArray &indices){
  LOG_DEBUG() << "Never use GridMPIOpenCL3D::GetAddress()!!\n";
  PSAbort(1);

  return 0;
}

// Assumes the dst address is in the device memory
void GridMPIOpenCL3D::Copyin(void *dst, const void *src, size_t size)
{
  LOG_DEBUG() << "Never use GridMPIOpen3D::Copyin()!!\n";
  PSAbort(1);
}

void GridMPIOpenCL3D::Copyin_CL(cl_mem dst_clmem, size_t offset, const void *src, size_t size) {
  cl_int status = 0;
  cl_command_queue queue = grid_clinfo_->get_queue();
  PSAssert(queue);
  status = clEnqueueWriteBuffer(
    queue, dst_clmem, CL_TRUE, /* block */
    offset, size, src,
    0, NULL, NULL);
  if (status != CL_SUCCESS) {
    LOG_DEBUG() << "Calling clEnqueueWriteBuffer() failed.\n";
  }
}

void GridMPIOpenCL3D::Copyin_CL(cl_mem dst_clmem, const void *src, size_t size) {
  Copyin_CL(dst_clmem, 0, src, size);
}


void GridMPIOpenCL3D::Set(const IntArray &indices, const void *buff)
{
  // The original implementation is
  // Copyin(GetAddress(indices), buf, elm_size());
  // However we can't use Copyin and GetAddress(indices) cannot be
  // used, either.
  cl_mem dst_clmem;
  size_t getaddress_ret;
  GetAddress_CL(indices, &dst_clmem, &getaddress_ret);
  Copyin_CL(dst_clmem, getaddress_ret, buff, elm_size());
}

// Assumes the source address is in the device memory
void GridMPIOpenCL3D::Copyout(void *dst, const void *src, size_t size)
{
  LOG_DEBUG() << "Never use GridMPIOpen3D::Copyout()!!\n";
  PSAbort(1);
}

void GridMPIOpenCL3D::Copyout_CL(void *dst, size_t offset, const cl_mem src_clmem, size_t size) {
  cl_int status = 0;
  cl_command_queue queue = grid_clinfo_->get_queue();
  PSAssert(queue);
  status = clEnqueueReadBuffer(
    queue, src_clmem, CL_TRUE, /* block */
    offset, size, dst,
    0, NULL, NULL);
  if (status != CL_SUCCESS) {
    LOG_DEBUG() << "Calling clEnqueueReadBuffer() failed.\n";
  }
}

void GridMPIOpenCL3D::Copyout_CL(void *dst, const cl_mem src_clmem, size_t size) {
  Copyout_CL(dst, 0, src_clmem, size);
}

void GridMPIOpenCL3D::Get(const IntArray &indices, void *buf) {
  // The original implementation is
  // Copyout(buf, GetAddress(indices), elm_size());
  // However we can't use Copyout and GetAddress(indices) cannot be
  // used, either.
  cl_mem src_clmem;
  size_t getaddress_ret;
  GetAddress_CL(indices, &src_clmem, &getaddress_ret);
  Copyout_CL(buf, getaddress_ret, src_clmem, elm_size());
}

std::ostream &GridMPIOpenCL3D::Print(std::ostream &os) const {
  os << "GridMPIOpenCL {"
     << "elm_size: " << elm_size_
     << ", size: " << size_
     << ", global offset: " << global_offset_
     << ", local offset: " << local_offset_
     << ", local size: " << local_size_
     << "}";
  return os;
}

// TODO: Not read in detail yet
// TODO: However, perhaps no problem
void GridMPIOpenCL3D::CopyoutHalo(int dim, unsigned width, bool fw,
                                bool diagonal) {
  PSAssert(num_dims() <= 3);
  halo_has_diagonal_ = diagonal;

  int fw_idx = (fw) ? 1: 0;

  IntArray halo_offset;
  if (!fw) halo_offset[dim] = local_size_[dim] - width;
  IntArray halo_size = local_size_;
  halo_size[dim] = width;
  size_t linear_size = halo_size.accumulate(num_dims());

  // First, copy out of OpenCL device memory to OpenCL pinned host memory
#if 0 // USE_MAPPED
  BufferOpenCLHostMapped *halo_opencl_host = halo_self_opencl_[dim][fw_idx];    
#else
  BufferOpenCLHost *halo_opencl_host = halo_self_opencl_[dim][fw_idx];  
#endif
  BufferHost *halo_mpi_host = halo_self_mpi_[dim][fw_idx];
  halo_opencl_host->EnsureCapacity(linear_size);
  static_cast<BufferOpenCLDev3D*>(buffer())->Copyout(
      *halo_opencl_host, halo_offset, halo_size, local_size_);

  // Next, copy out to the halo buffer from OpenCL pinned host memory
  halo_mpi_host->EnsureCapacity(CalcHaloSize(dim, width, diagonal));
  if (dim == num_dims_ - 1 || !diagonal) {
    halo_opencl_host->Copyout(*halo_mpi_host, halo_size.accumulate(num_dims_));
    return;
  }

  switch (num_dims_) {
    case 2:
      //CopyoutHalo2D0(width, fw, *halo_buf);
      PSAbort(1); // TODO: not supported
      break;
    case 3:
      if (dim == 0) {
        CopyoutHalo3D0(width, fw);
      } else if (dim == 1) {
        CopyoutHalo3D1(width, fw);
      } else {
        LOG_ERROR() <<
            "This case should have been handled in the first block"
            " of this function.\n";
        PSAbort(1);
      }
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << num_dims_ << "\n";
      PSAbort(1);
  }
}

// REFACTORING: This is almost identical to the parent class implementation. 
void GridMPIOpenCL3D::CopyoutHalo3D1(unsigned width, bool fw) {
  int nd = 3;
  int fw_idx = fw ? 1 : 0;
  // copy diag
  char *buf = (char*)halo_self_mpi_[1][fw_idx]->Get();
      // halo_self_mpi_ is BufferHost
  IntArray sg_offset;
  if (!fw) sg_offset[1] = local_size_[1] - width;
  IntArray sg_size(local_size_[0], width, halo_bw_width_[2]);
  IntArray halo_size(local_size_[0], local_size_[1],
                     halo_bw_width_[2]);
  // different  
  // halo_peer_opencl_ is BufferOpenCLHost
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_opencl_[2][0]->Get(),
                 halo_size, buf, sg_offset, sg_size);
  buf += sg_size.accumulate(nd) * elm_size_;

  // copy halo
  sg_size[2] = local_size_[2];
  // different
  // halo_self_opencl_ is BufferOpenCLHost{,Mapped}
  halo_self_opencl_[1][fw_idx]->Copyout(buf, sg_size.accumulate(nd));
  buf += sg_size.accumulate(nd) * elm_size_;
  
  // copy diag
  sg_size[2] = halo_fw_width_[2];
  halo_size[2] = halo_fw_width_[2];
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_opencl_[2][1]->Get(),
                 halo_size, buf, sg_offset, sg_size);
}

// REFACTORING: This is almost identical to the parent class implementation. 
//void GridMPIOpenCL3D::CopyoutHalo3D0(unsigned width, int fw) {
void GridMPIOpenCL3D::CopyoutHalo3D0(unsigned width, bool fw) {
  int fw_idx = fw ? 1 : 0;
  // halo_self_mpi_ is BufferHost
  char *buf = (char*)halo_self_mpi_[0][fw_idx]->Get();
  size_t line_size = elm_size_ * width;  
  size_t xoffset = fw ? 0 : local_size_[0] - width;
  // halo_peer_opencl_ is BufferOpenCLHost
  char *halo_bw_1 = (char*)halo_peer_opencl_[1][0]->Get() + xoffset * elm_size_;
  char *halo_fw_1 = (char*)halo_peer_opencl_[1][1]->Get() + xoffset * elm_size_;
  char *halo_bw_2 = (char*)halo_peer_opencl_[2][0]->Get() + xoffset * elm_size_;
  char *halo_fw_2 = (char*)halo_peer_opencl_[2][1]->Get() + xoffset * elm_size_;
  // halo_self_opencl_ is BufferOpenCLHost{,Mapped}
  //char *halo_0 = data_[0] + xoffset * elm_size_;
  char *halo_0 = (char*)halo_self_opencl_[0][fw_idx]->Get();
  for (int k = 0; k < halo_bw_width_[2]+local_size_[2]+halo_fw_width_[2]; ++k) {
    // copy from the 2nd-dim backward halo 
    char *t = halo_bw_1;
    for (int j = 0; j < halo_bw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    }
    halo_bw_1 += local_size_[0] * halo_bw_width_[1] * elm_size_;

    char **h;
    size_t halo_x_size;
    // select halo source
    if (k < halo_bw_width_[2]) {
      h = &halo_bw_2;
      halo_x_size = local_size_[0] * elm_size_;
    } else if (k < halo_bw_width_[2] + local_size_[2]) {
      h = &halo_0;
      halo_x_size = elm_size_ * width;
    } else {
      h = &halo_fw_2;
      halo_x_size = local_size_[0] * elm_size_;
    }
    t = *h;
    for (int j = 0; j < local_size_[1]; ++j) {
      memcpy(buf, t, line_size);
      buf += line_size;
      t += halo_x_size;
    }
    *h += halo_x_size * local_size_[1];

    // copy from the 2nd-dim forward halo 
    t = halo_fw_1;
    for (int j = 0; j < halo_fw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    }
    halo_fw_1 += local_size_[0] * halo_fw_width_[1] * elm_size_;
  }
}

void GridMPIOpenCL3D::FixupBufferPointers() {
  LOG_VERBOSE() << "Fixup buffer cl_mem\n";
  if (data_buffer_[0]) {
    if (data_[0] != 
        reinterpret_cast<char *>
          (static_cast<BufferOpenCLDev*>(data_buffer_[0])->Get_buf_mem()))
    {
      LOG_DEBUG() << "Buf pointer updated by " << "data_buffer_[0]->Get_buf_mem()" << "\n";
    }
    // FIXME
    // FIXME: Be very careful!!
    data_[0] = reinterpret_cast<char *>
                  (static_cast<BufferOpenCLDev*>(data_buffer_[0])->Get_buf_mem());
    data_[1] = reinterpret_cast<char *>
                  (static_cast<BufferOpenCLDev*>(data_buffer_[1])->Get_buf_mem());
  } else {
    LOG_VERBOSE() << "data buffer 0\n";
    data_[0] = 0;
    data_[1] = 0;
  }

  dev_.p0 = data_[0];
  dev_.diag = halo_has_diagonal();
  LOG_VERBOSE() << "Diag: " << dev_.diag << "\n";
  if (data_buffer_[0]) {
    dev_.pitch = static_cast<BufferOpenCLDev*>(buffer())
        ->GetPitch() / elm_size();
    LOG_DEBUG() << "Pitch: " << dev_.pitch << "\n";
    for (int i = 0; i < num_dims(); ++i) {
      dev_.dim[i]  = size()[i];
      dev_.local_size[i] = local_size()[i];
      dev_.local_offset[i] = local_offset()[i];      
      // halo_peer_dev_ is BufferOpenCLDev
      clmem_halo_peer_fw_[i] = halo_peer_dev_[i][1]->Get_buf_mem();
      clmem_halo_peer_bw_[i] = halo_peer_dev_[i][0]->Get_buf_mem();

      for (int j = 0; j < 2; ++j) {
        dev_.halo[i][j] = halo_peer_dev_[i][j]->Get_buf_mem();
      }
      dev_.halo_width[i][1] = halo_fw_width()[i];
      dev_.halo_width[i][0] = halo_bw_width()[i];
    }
  }
}

void GridMPIOpenCL3D::EnsureRemoteGrid(
                const IntArray &local_offset,
                const IntArray &local_size)
{
  if (remote_grid_ == NULL) {
    // For sure
    PSAssert(grid_clinfo_);
    remote_grid_ = GridMPIOpenCL3D::Create(
            type_, elm_size_, num_dims_,
            size_, false, global_offset_,
            local_offset, local_size,
              0,
            grid_clinfo_);
  } else {
    remote_grid_->Resize(local_offset, local_size);
  }
}



//
// Grid Space
//

GridSpaceMPIOpenCL::GridSpaceMPIOpenCL(
    int num_dims, const IntArray &global_size,
    int proc_num_dims, const IntArray &proc_size, int my_rank,
    CLbaseinfo *clinfo_in):
    GridSpaceMPI(num_dims, global_size, proc_num_dims,
                 proc_size, my_rank),
    space_clinfo_(clinfo_in)
{
  buf = new BufferHost(1);
}


GridSpaceMPIOpenCL::~GridSpaceMPIOpenCL() {
  delete buf;
  FOREACH (it, load_neighbor_prof_.begin(), load_neighbor_prof_.end()) {
    delete[] it->second;
  }
}

GridMPIOpenCL3D *GridSpaceMPIOpenCL::CreateGrid(
    PSType type, int elm_size, int num_dims, const IntArray &size,
    bool double_buffering, const IntArray &global_offset,
    int attr,
    CLbaseinfo *clinfo)
{
  IntArray grid_size = size;
  IntArray grid_global_offset = global_offset;
  if (num_dims_ < num_dims) {
    // The grid space has smaller dimension.
    LOG_ERROR() << "Cannot create " << num_dims << "-d grid in "
                << num_dims_ << "-d space.\n";
    PSAbort(1);
  } else if (num_dims_ > num_dims) {
    // Adapt the dimension of grid to the grid space
    for (int i = num_dims; i < num_dims_; ++i) {
      grid_size[i] = 1;
      grid_global_offset[i] = 0;
    }
    num_dims = num_dims_;
  }

  IntArray local_offset, local_size;  
  PartitionGrid(num_dims, grid_size, grid_global_offset,
                local_offset, local_size);

  LOG_DEBUG() << "local_size: " << local_size << "\n";
  GridMPIOpenCL3D *g = GridMPIOpenCL3D::Create(
                                          type, elm_size, num_dims, grid_size,
                                          double_buffering,
                                          grid_global_offset, local_offset,
                                          local_size, attr,
                                          clinfo);
  LOG_DEBUG() << "grid created\n";
  RegisterGrid(g);
  LOG_DEBUG() << "grid registered\n";  
  DataCopyProfile *profs = new DataCopyProfile[num_dims * 2];
  load_neighbor_prof_.insert(make_pair(g->id(), profs));
  return g;
}

GridMPIOpenCL3D *GridSpaceMPIOpenCL::CreateGrid(
    PSType type, int elm_size, int num_dims, const IntArray &size,
    bool double_buffering, const IntArray &global_offset,
    int attr)
{
  PSAssert(space_clinfo_ != NULL);
  return CreateGrid(
      type, elm_size, num_dims, size, double_buffering,
      global_offset, attr, space_clinfo_);
}

// Jut copy out halo from GPU memory
void GridSpaceMPIOpenCL::ExchangeBoundariesStage1(
    GridMPI *g, int dim, unsigned halo_fw_width,
    unsigned halo_bw_width, bool diagonal,
    bool periodic ) const {

  GridMPIOpenCL3D *grid = static_cast<GridMPIOpenCL3D*>(g);
  if (grid->empty_) return;

  LOG_DEBUG() << "Periodic grid?: " << periodic << "\n";

  DataCopyProfile *profs = find<int, DataCopyProfile*>(
      load_neighbor_prof_, g->id(), NULL);
  DataCopyProfile &prof_upw = profs[dim*2];
  DataCopyProfile &prof_dwn = profs[dim*2+1];  
  Stopwatch st;
  // Sends out the halo for backward access

  if (halo_bw_width > 0 &&
      (periodic ||
       grid->local_offset_[dim] + grid->local_size_[dim] < grid->size_[dim])) {
#if defined(PS_VERBOSE)            
    LOG_DEBUG() << "Sending halo of " << bw_size << " elements"
              << " for bw access to " << fw_peer << "\n";
#endif    
    st.Start();
    grid->CopyoutHalo(dim, halo_bw_width, false, diagonal);
    prof_upw.gpu_to_cpu += st.Stop();
  }
  
  // Sends out the halo for forward access
  if (halo_fw_width > 0 &&
      (periodic || grid->local_offset_[dim] > 0)) {
#if defined(PS_VERBOSE)            
    LOG_DEBUG() << "Sending halo of " << fw_size << " elements"
                << " for fw access to " << bw_peer << "\n";
#endif    
    st.Start();    
    grid->CopyoutHalo(dim, halo_fw_width, true, diagonal);
    prof_dwn.gpu_to_cpu += st.Stop();
  }
}

// Exchanges halos on CPU memory, and copy in the new halo to GPU
// memory. Assumes the new halo to exchange is already copied out of
// GPU memory at Stage 1.
// Note: width is unsigned. 
void GridSpaceMPIOpenCL::ExchangeBoundariesStage2(
    GridMPI *g, int dim, unsigned halo_fw_width,
    unsigned halo_bw_width, bool diagonal,
    bool periodic) const {

  GridMPIOpenCL3D *grid = static_cast<GridMPIOpenCL3D*>(g);
  if (grid->empty_) return;

  int fw_peer = fw_neighbors_[dim];
  int bw_peer = bw_neighbors_[dim];
  ssize_t fw_size = grid->CalcHaloSize(dim, halo_fw_width, diagonal);
  ssize_t bw_size = grid->CalcHaloSize(dim, halo_bw_width, diagonal);
  LOG_DEBUG() << "Periodic grid?: " << periodic << "\n";

  DataCopyProfile *profs = find<int, DataCopyProfile*>(
      load_neighbor_prof_, g->id(), NULL);
  DataCopyProfile &prof_upw = profs[dim*2];
  DataCopyProfile &prof_dwn = profs[dim*2+1];  
  Stopwatch st;

  bool req_bw_active = false;
  bool req_fw_active = false;
  MPI_Request req_bw;
  MPI_Request req_fw;
  
  if (halo_bw_width > 0 &&
      (periodic ||
       grid->local_offset_[dim] + grid->local_size_[dim] < grid->size_[dim])) {
    st.Start();
    grid->halo_self_mpi_[dim][0]->MPIIsend(fw_peer, comm_, &req_bw,
                                           IntArray(), IntArray(bw_size));
    prof_upw.cpu_out += st.Stop();
    req_bw_active = true;
  }
  
  // Sends out the halo for forward access
  if (halo_fw_width > 0 &&
      (periodic || grid->local_offset_[dim] > 0)) {
    st.Start();        
    grid->halo_self_mpi_[dim][1]->MPIIsend(bw_peer, comm_, &req_fw,
                                           IntArray(), IntArray(fw_size));
    prof_dwn.cpu_out += st.Stop();
    req_fw_active = true;
  }

  // Receiving halo for backward access
  RecvBoundaries(grid, dim, halo_bw_width, false, diagonal,
                 periodic, bw_size, prof_dwn);
  
  // Receiving halo for forward access
  RecvBoundaries(grid, dim, halo_fw_width, true, diagonal,
                 periodic, fw_size, prof_upw);

  // Ensure the exchanges are done. Otherwise, the send buffers can be
  // overwritten by the next iteration.
  if (req_fw_active) CHECK_MPI(MPI_Wait(&req_fw, MPI_STATUS_IGNORE));
  if (req_bw_active) CHECK_MPI(MPI_Wait(&req_bw, MPI_STATUS_IGNORE));

  grid->FixupBufferPointers();
  return;
}

bool GridSpaceMPIOpenCL::SendBoundaries(GridMPIOpenCL3D *grid, int dim,
                                      unsigned width,
                                      bool forward, bool diagonal,
                                      bool periodic,
                                      ssize_t halo_size,
                                      DataCopyProfile &prof,
                                      MPI_Request &req) const {
  // Nothing to do since the width is zero
  if (width <= 0) {
    return false;
  }

  // Do nothing if this process is on the end of the dimension and
  // periodic boundary is not set.
  if (!periodic) {
    if ((!forward &&
         grid->local_offset_[dim] + grid->local_size_[dim] == grid->size_[dim]) ||
        (forward && grid->local_offset_[dim] == 0)) {
      return false;
    }
  }
  
  int dir_idx = forward ? 1 : 0;
  int peer = forward ? bw_neighbors_[dim] : fw_neighbors_[dim];  
  LOG_DEBUG() << "Sending halo of " << halo_size << " elements"
              << " for access to " << peer << "\n";
  Stopwatch st;  
  st.Start();
  grid->CopyoutHalo(dim, width, forward, diagonal);
  prof.gpu_to_cpu += st.Stop();
#if defined(PS_VERBOSE)
  LOG_VERBOSE() << "opencl ->";
  // halo_self_opencl_ is BufferOpenCLHost
  grid->halo_self_opencl_[dim][dir_idx]->print<float>(std::cerr);
  LOG_VERBOSE() << "host ->";
  // halo_self_mpi_ is BufferHost
  grid->halo_self_mpi_[dim][dir_idx]->print<float>(std::cerr);
#endif

  st.Start();
  // halo_self_mpi_ is BufferHost
  grid->halo_self_mpi_[dim][dir_idx]->MPIIsend(peer, comm_, &req,
                                               IntArray(),
                                               IntArray(halo_size));
  prof.cpu_out += st.Stop();
  return true;
}

bool GridSpaceMPIOpenCL::RecvBoundaries(GridMPIOpenCL3D *grid, int dim,
                                      unsigned width,
                                      bool forward, bool diagonal,
                                      bool periodic, ssize_t halo_size,
                                      DataCopyProfile &prof) const {
  int peer = forward ? fw_neighbors_[dim] : bw_neighbors_[dim];
  int dir_idx = forward ? 1 : 0;
  bool is_last_process =
      grid->local_offset_[dim] + grid->local_size_[dim]
      == grid->size_[dim];
  bool is_first_process = grid->local_offset_[dim] == 0;
  Stopwatch st;

  if (width == 0 ||
      (!periodic && ((forward && is_last_process) ||
                        (!forward && is_first_process)))) {
    if (forward) {
      grid->halo_fw_width_[dim] = 0;
      grid->halo_fw_size_[dim].assign(0);
    } else {
      grid->halo_bw_width_[dim] = 0;
      grid->halo_bw_size_[dim].assign(0);
    }
    return false;
  }

#if defined(PS_DEBUG)
  if (!periodic) {
    if( (forward && grid->local_offset_[dim] +
        grid->local_size_[dim] + width
         > grid->size_[dim]) ||
        (!forward && grid->local_offset_[dim] - width < 0)
        ) {
      LOG_ERROR() << "Off limit accesses: "
                  << "local_offset: " << grid->local_offset_[dim]
                  << ", local_size: " << grid->local_size_[dim]
                  << ", width: " << width
                  << ", grid size: " << grid->size_[dim]
                  << "\n";
      LOG_ERROR() << "is_first: " << is_first_process
                  << ", is_last: " << is_last_process
                  << ", forward: " << forward
                  << "\n";
      PSAbort(1);
    }
  }
#endif  
  LOG_DEBUG() << "Receiving halo of " << halo_size
              << " bytes from " << peer << "\n";
  
  // halo_peer_opencl_ is BufferOpenCLHost
  grid->halo_peer_opencl_[dim][dir_idx]->EnsureCapacity(halo_size);
  if (forward) {
    grid->halo_fw_width_[dim] = width;
  } else {
    grid->halo_bw_width_[dim] = width;
  }
  grid->SetHaloSize(dim, forward, width, diagonal);
  st.Start();
  // halo_peer_opencl_ is BufferOpenCLHost
  grid->halo_peer_opencl_[dim][dir_idx]->
      Buffer::MPIRecv(peer, comm_, IntArray(halo_size));
  prof.cpu_in += st.Stop();
  st.Start(); 
  // halo_peer_dev_ is BufferOpenCLDev
  // halo_peer_opencl_ is BufferOpenCLHost
  grid->halo_peer_dev_[dim][dir_idx]->
      Copyin(*grid->halo_peer_opencl_[dim][dir_idx], IntArray(),
             IntArray(halo_size));
  prof.cpu_to_gpu += st.Stop();
  return true;
}

// Note: width is unsigned. 
void GridSpaceMPIOpenCL::ExchangeBoundaries(
    GridMPI *g, int dim, unsigned halo_fw_width,
    unsigned halo_bw_width, bool diagonal, bool periodic) const {

  GridMPIOpenCL3D *grid = static_cast<GridMPIOpenCL3D*>(g);
  if (grid->empty_) return;

  ssize_t fw_size = grid->CalcHaloSize(dim, halo_fw_width, diagonal);
  ssize_t bw_size = grid->CalcHaloSize(dim, halo_bw_width, diagonal);
  LOG_DEBUG() << "Periodic grid?: " << periodic << "\n";

  DataCopyProfile *profs = find<int, DataCopyProfile*>(
      load_neighbor_prof_, g->id(), NULL);
  DataCopyProfile &prof_upw = profs[dim*2];
  DataCopyProfile &prof_dwn = profs[dim*2+1];  
  MPI_Request req_bw, req_fw;

  // Sends out the halo for backward access
  bool req_bw_active =
      SendBoundaries(grid, dim, halo_bw_width, false, diagonal,
                     periodic, bw_size, prof_upw, req_bw);
  
  // Sends out the halo for forward access
  bool req_fw_active =
      SendBoundaries(grid, dim, halo_fw_width, true, diagonal,
                     periodic, fw_size, prof_dwn, req_fw);

  // Receiving halo for backward access
  RecvBoundaries(grid, dim, halo_bw_width, false, diagonal,
                 periodic, bw_size, prof_dwn);
  
  // Receiving halo for forward access
  RecvBoundaries(grid, dim, halo_fw_width, true, diagonal,
                 periodic, fw_size, prof_upw);

  // Ensure the exchanges are done. Otherwise, the send buffers can be
  // overwritten by the next iteration.
  if (req_fw_active) CHECK_MPI(MPI_Wait(&req_fw, MPI_STATUS_IGNORE));
  if (req_bw_active) CHECK_MPI(MPI_Wait(&req_bw, MPI_STATUS_IGNORE));

  grid->FixupBufferPointers();
  return;
}

void GridSpaceMPIOpenCL::HandleFetchRequest(GridRequest &req, GridMPI *g) {
  LOG_DEBUG() << "HandleFetchRequest\n";
  GridMPIOpenCL3D *gm = static_cast<GridMPIOpenCL3D*>(g);
  FetchInfo finfo;
  int nd = num_dims_;
  CHECK_MPI(MPI_Recv(&finfo, sizeof(FetchInfo), MPI_BYTE,
                     req.my_rank, 0, comm_, MPI_STATUS_IGNORE));
  size_t bytes = finfo.peer_size.accumulate(nd) * g->elm_size();
  // 
  // char *buf = (char*)halo_self_mpi_[1][fw_idx]->Get(); , for example
  buf->EnsureCapacity(bytes);
  static_cast<BufferOpenCLDev3D*>(gm->buffer())->Copyout(
      *buf, finfo.peer_offset - g->local_offset(), finfo.peer_size);
  SendGridRequest(my_rank_, req.my_rank, comm_, FETCH_REPLY);
  MPI_Request mr;
  buf->MPIIsend(req.my_rank, comm_, &mr, IntArray(), IntArray(bytes));
  //CHECK_MPI(PS_MPI_Isend(buf, bytes, MPI_BYTE, req.my_rank, 0, comm_, &mr));
}

void GridSpaceMPIOpenCL::HandleFetchReply(GridRequest &req, GridMPI *g,
                                        std::map<int, FetchInfo> &fetch_map,
                                        GridMPI *sg) {
  LOG_DEBUG() << "HandleFetchReply\n";
  GridMPIOpenCL3D *sgm = static_cast<GridMPIOpenCL3D*>(sg);
  const FetchInfo &finfo = fetch_map[req.my_rank];
  PSAssert(GetProcessRank(finfo.peer_index) == req.my_rank);
  size_t bytes = finfo.peer_size.accumulate(num_dims_) * g->elm_size();
  LOG_DEBUG() << "Fetch reply data size: " << bytes << "\n";
  // 
  // char *buf = (char*)halo_self_mpi_[1][fw_idx]->Get(); , for example
  buf->EnsureCapacity(bytes);
  buf->Buffer::MPIRecv(req.my_rank, comm_, IntArray(bytes));
  LOG_DEBUG() << "Fetch reply received\n";
  static_cast<BufferOpenCLDev3D*>(sgm->buffer())->Copyin(
      *buf, finfo.peer_offset - sg->local_offset(),
      finfo.peer_size);
  sgm->FixupBufferPointers();
  return;
}

inline index_t GridCalcOffset3D(const IntArray &index,
                                const IntArray &size,
                                size_t pitch) {
  return index[0] + index[1] * pitch + index[2] * pitch * size[1];
}

void GridMPIOpenCL3D::GetAddress_CL(
      const IntArray &indices_param,
      cl_mem *buf_clmem_ret,
      size_t *offset_ret
)
{
  // Anyway initialize
  *buf_clmem_ret = 0;
  *offset_ret = 0;

  // Use the remote grid if remote_grid_active is true.
  if (remote_grid_active()) {
    PSAbort(1);
  }
  
  IntArray indices = indices_param -local_offset();
  bool diag = halo_has_diagonal();
  for (int i = 0; i < num_dims(); ++i) {
    if (indices[i] < 0 || indices[i] >= local_size()[i]) {
      for (int j = i+1; j < PS_MAX_DIM; ++j) {
        if (diag) indices[i] += halo_bw_width()[i];
      }
      if (indices[i] < 0) {
        indices[i] += halo_bw_width()[i];
        *offset_ret = GridCalcOffset3D(indices, halo_bw_size()[i]);
        *buf_clmem_ret = _clmem_halo_peer_bw()[i];
      } else {
        indices[i] -= local_size()[i];
        *offset_ret = GridCalcOffset3D(indices, halo_fw_size()[i]);
        *buf_clmem_ret = _clmem_halo_peer_fw()[i];
      }
      *offset_ret *= elm_size();
      return;
    }
  }

  size_t pitch =  static_cast<BufferOpenCLDev3D*>(buffer())->GetPitch()
      / elm_size_;

  // char *_data() { return data_[0]; }
  // data_[0] = reinterpret_cast<char *>
  //    (static_cast<BufferOpenCLDev*>(data_buffer_[0])->Get_buf_mem());
  // FIXME
  // FIXME: Very careful!!
  *buf_clmem_ret = reinterpret_cast<cl_mem>(_data());
  *offset_ret = GridCalcOffset3D(indices, local_size(), pitch) * elm_size_;
  return;

}

GridMPI *GridSpaceMPIOpenCL::LoadNeighbor(
    GridMPI *g,
    const IntArray &halo_fw_width,
    const IntArray &halo_bw_width,
    bool diagonal,  bool reuse, bool periodic,
    const bool *fw_enabled,  const bool *bw_enabled,
    CLbaseinfo *cl_stream) {
  
  // set the stream of buffer by the stream parameter
  GridMPIOpenCL3D *gmc = static_cast<GridMPIOpenCL3D*>(g);
  gmc->SetOpenCLinfo(cl_stream);
  GridMPI *rg = GridSpaceMPI::LoadNeighbor(g, halo_fw_width, halo_bw_width,
                                           diagonal, reuse, periodic, fw_enabled,
                                           bw_enabled);
  gmc->SetOpenCLinfo(0);
  return rg;
}

GridMPI *GridSpaceMPIOpenCL::LoadNeighborStage1(
    GridMPI *g,
    const IntArray &halo_fw_width,
    const IntArray &halo_bw_width,
    bool diagonal,  bool reuse, bool periodic,
    const bool *fw_enabled,  const bool *bw_enabled,
    CLbaseinfo *cl_stream)
{
  
  // set the stream of buffer by the stream parameter
  GridMPIOpenCL3D *gmc = static_cast<GridMPIOpenCL3D *>(g);
  gmc->SetOpenCLinfo(cl_stream);
  IntArray halo_fw_width_tmp(halo_fw_width);
  halo_fw_width_tmp.SetNoLessThan(0);
  IntArray halo_bw_width_tmp(halo_bw_width);
  halo_bw_width_tmp.SetNoLessThan(0);
  //for (int i = g->num_dims_ - 1; i >= 0; --i) {
  for (int i = g->num_dims_ - 2; i >= 0; --i) {  
    LOG_VERBOSE() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundariesStage1(g, i, halo_fw_width[i],
                             halo_bw_width[i], diagonal,
                             periodic);
  }
  gmc->SetOpenCLinfo(0);
  return NULL;
}

GridMPI *GridSpaceMPIOpenCL::LoadNeighborStage2(
    GridMPI *g,
    const IntArray &halo_fw_width,
    const IntArray &halo_bw_width,
    bool diagonal,  bool reuse, bool periodic,
    const bool *fw_enabled,  const bool *bw_enabled,
    CLbaseinfo *cl_stream) {
  
  // set the stream of buffer by the stream parameter
  GridMPIOpenCL3D *gmc = static_cast<GridMPIOpenCL3D*>(g);
  gmc->SetOpenCLinfo(cl_stream);
  IntArray halo_fw_width_tmp(halo_fw_width);
  halo_fw_width_tmp.SetNoLessThan(0);
  IntArray halo_bw_width_tmp(halo_bw_width);
  halo_bw_width_tmp.SetNoLessThan(0);

  // Does not perform staging for the continuous dimension. 
  int i = g->num_dims_ - 1;
  ExchangeBoundaries(g, i, halo_fw_width[i],
                     halo_bw_width[i], diagonal, periodic);

  // For the non-continuous dimensions, finish the exchange steps 
  for (int i = g->num_dims_ - 2; i >= 0; --i) {
    LOG_VERBOSE() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundariesStage2(g, i, halo_fw_width[i],
                             halo_bw_width[i], diagonal, periodic);
  }
  gmc->SetOpenCLinfo(0);
  return NULL;
}


std::ostream& GridSpaceMPIOpenCL::PrintLoadNeighborProf(std::ostream &os) const {
  StringJoin sj("\n");
  FOREACH (it, load_neighbor_prof_.begin(), load_neighbor_prof_.end()) {
    int grid_id = it->first;
    DataCopyProfile *profs = it->second;
    StringJoin sj_grid;
    for (int i = 0; i < num_dims_*2; i+=2) {
      sj_grid << "upw: " << profs[i] << ", dwn: " << profs[i+1];
    }
    sj << grid_id << ": " << sj_grid.str();
  }
  return os << sj.str() << "\n";
}


} // namespace runtime
} // namespace physis

