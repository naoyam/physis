#include "runtime/grid_mpi_openmp.h"

#include <limits.h>
#include "runtime/grid_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_wrapper.h"
#include "runtime/grid_util_mpi_openmp.h"

using namespace std;

namespace physis {
namespace runtime {

GridMPIOpenMP::GridMPIOpenMP(PSType type, int elm_size, int num_dims,
                             const IntArray &size,
                             bool double_buffering, const IntArray &global_offset,
                             const IntArray &local_offset,                 
                             const IntArray &local_size,
                             const IntArray &division,
                             int attr):
    GridMPI(type, elm_size, num_dims,
            size,
            double_buffering, global_offset,
            local_offset,                 
            local_size, attr),
    division_(division)
{
  data_MP_[0] = 0;
  data_MP_[1] = 0;
  data_[0] = 0;
  data_[1] = 0;
  
}

// This is the only interface to create a GridMPI grid
GridMPIOpenMP *GridMPIOpenMP::Create(
    PSType type, int elm_size,
    int num_dims, const IntArray &size,
    bool double_buffering,
    const IntArray &global_offset,
    const IntArray &local_offset,
    const IntArray &local_size,
    const IntArray &division,
    int attr) {
  GridMPIOpenMP *gm = new GridMPIOpenMP(
      type, elm_size, num_dims, size,
      double_buffering, global_offset,
      local_offset, local_size, division,
      attr);
  gm->InitBuffer();
  return gm;
}

void GridMPIOpenMP::InitBuffer() {

  BufferHostOpenMP *data_buffer_mp0 =
      new BufferHostOpenMP(num_dims_, elm_size_, division_);
  BufferHostOpenMP *data_buffer_mp1 = 0;

  data_buffer_mp0->Allocate(local_size_);
  if (double_buffering_) {
    data_buffer_mp1 = new BufferHostOpenMP(num_dims_, elm_size_, division_);
    data_buffer_mp1->Allocate(local_size_);    
  } else {
    data_buffer_mp1 = data_buffer_mp0;
  }

  data_MP_[0] = (char **)data_buffer_mp0->Get_MP();
  data_MP_[1] = (char **)data_buffer_mp1->Get_MP();

  data_buffer_[0] = data_buffer_mp0;
  data_buffer_[1] = data_buffer_mp1;
}


void GridMPIOpenMP::Swap() {
  std::swap(data_[0], data_[1]);
  std::swap(data_buffer_[0], data_buffer_[1]);
  std::swap(data_MP_[0], data_MP_[1]);
}

void GridMPIOpenMP::EnsureRemoteGrid(const IntArray &local_offset,
                                     const IntArray &local_size) {
  if (remote_grid_ == NULL) {
    remote_grid_ = GridMPIOpenMP::Create(
        type_, elm_size_, num_dims_,
        size_, false, global_offset_,
        local_offset, local_size, 
        division_,
        0);
  } else {
    remote_grid_->Resize(local_offset, local_size);
  }
}

// Copy out halo with diagonal points
// PRECONDITION: halo for the first dim is already exchanged
void GridMPIOpenMP::CopyoutHalo2D0(unsigned width, bool fw, char *buf) {

  // buf SHOULD BE not within BufferHostOpenMP
  // halo_peer_bw_ is not within BufferHostOpenMP
  // halo_peer_fw_ is not within BufferHostOpenMP

  LOG_DEBUG() << "2d copyout halo\n";
  size_t line_size = width * elm_size_;
  // copy diag
  size_t offset = fw ? 0 : (local_size_[0] - width) * elm_size_;
  char *diag_buf = halo_peer_bw_[1] + offset;
  for (unsigned i = 0; i < halo_bw_width_[1]; ++i) {
    memcpy(buf, diag_buf, line_size);
    buf += line_size;
    diag_buf += local_size_[0] * elm_size_;
  }
  
  // copy halo
  IntArray sg_offset;
  if (!fw) sg_offset[0] = local_size_[0] - width;
  IntArray sg_size(width, local_size_[1]);

  // data_[0] cannot be used, so
  // need modification
#if 0
  CopyoutSubgrid(elm_size_, 2, data_[0], local_size_,
                 buf, sg_offset, sg_size);
#endif
  // As data_[0] should be (char*)data_buffer_[0]->Get();
  // And data_buffer_ should be BufferHost(OpenMP):
  BufferHostOpenMP* data_buffer_mp0 = 
      dynamic_cast<BufferHostOpenMP*>(data_buffer_[0]);
  PSAssert(data_buffer_mp0);
  CopyinoutSubgrid(
      1, elm_size_, 2,
      data_buffer_mp0,
      local_size_, buf, sg_offset, sg_size);

  buf += line_size * local_size_[1];

  // copy diag
  diag_buf = halo_peer_fw_[1] + offset;
  for (unsigned i = 0; i < halo_fw_width_[1]; ++i) {
    memcpy(buf, diag_buf, line_size);
    buf += line_size;
    diag_buf += local_size_[0] * elm_size_;
  }
  
  return;
}

void GridMPIOpenMP::CopyoutHalo3D0(unsigned width, bool fw, char *buf) {
  // buf SHOULD BE not within BufferHostOpenMP
  // halo_peer_fw is locally malloc'ed space, not in BufferHostOpenMP
  // halo_peer_bw is locally malloc'ed space, not in BufferHostOpenMP
  // halo_0 needs special treatment

  size_t line_size = elm_size_ * width;  
  size_t xoffset = fw ? 0 : local_size_[0] - width;
  char *halo_bw_1 = halo_peer_bw_[1] + xoffset * elm_size_;
  char *halo_fw_1 = halo_peer_fw_[1] + xoffset * elm_size_;
  char *halo_bw_2 = halo_peer_bw_[2] + xoffset * elm_size_;
  char *halo_fw_2 = halo_peer_fw_[2] + xoffset * elm_size_;
#if 0
  // This method cannot be used, sorry
  char *halo_0 = data_[0] + xoffset * elm_size_;
#endif
  BufferHostOpenMP* data_buffer_mp0 = 
      dynamic_cast<BufferHostOpenMP*>(data_buffer_[0]);
  PSAssert(data_buffer_mp0);
  IntArray emptyoffset(0,0,0);

  for (unsigned k = 0; k < halo_bw_width_[2]+local_size_[2]+halo_fw_width_[2]; ++k) {
    // copy from the 2nd-dim backward halo 
    char *t = halo_bw_1;
    bool using_halo_0_p = 0;

    for (unsigned j = 0; j < halo_bw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    }
    halo_bw_1 += local_size_[0] * halo_bw_width_[1] * elm_size_;

    char **h;
    // select halo source
    if (k < halo_bw_width_[2]) {
      h = &halo_bw_2;
    } else if (k < halo_bw_width_[2] + local_size_[2]) {
#if 0
      h = &halo_0;
#else
      h = 0;
#endif
      using_halo_0_p = 1; // special treatment
    } else {
      h = &halo_fw_2;
    }

    if (h) 
      t = *h;
    else
      t = 0;

    for (int j = 0; j < local_size_[1]; ++j) {
      if (!using_halo_0_p) {
        memcpy(buf, t, line_size);
      } else {
        //memcpy(buf, t, line_size);
        size_t mp0_offset = 
            xoffset + j * local_size_[0] +
            (k - halo_bw_width_[2]) * local_size_[0] * local_size_[1];
        IntArray want_size(width, 1, 1);
        data_buffer_mp0->Copyout(
            buf, emptyoffset, want_size, mp0_offset);

      }
      buf += line_size;
      if (t) t += local_size_[0] * elm_size_;

    } // for (int j = 0; j < local_size_[1]; ++j)

    if (h) *h += local_size_[0] * local_size_[1] * elm_size_;

    // copy from the 2nd-dim forward halo 
    t = halo_fw_1;
    for (unsigned j = 0; j < halo_fw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    } // for (int j = 0; j < halo_fw_width_[1]; j++)
    halo_fw_1 += local_size_[0] * halo_fw_width_[1] * elm_size_;

  } // for (int k = 0; k < halo_bw_width_[2]+local_size_[2]+halo_fw_width_[2]; ++k)
  
}

void GridMPIOpenMP::CopyoutHalo3D1(unsigned width, bool fw, char *buf) {
  int nd = 3;
  // copy diag
  IntArray sg_offset;
  if (!fw) sg_offset[1] = local_size_[1] - width;
  IntArray sg_size(local_size_[0], width, halo_bw_width_[2]);
  IntArray halo_size(local_size_[0], local_size_[1],
                     halo_bw_width_[2]);
  // different
  // This should be safe
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_bw_[2],
                 halo_size, buf, sg_offset, sg_size);
  buf += sg_size.accumulate(nd) * elm_size_;

  // copy halo
  // Using data_[0] originally, need change
  // As data_[0] should be (char*)data_buffer_[0]->Get();
  // And data_buffer_ should be BufferHost(OpenMP):
  sg_size[2] = local_size_[2];
  BufferHostOpenMP* data_buffer_mp0 = 
      dynamic_cast<BufferHostOpenMP*>(data_buffer_[0]);
  PSAssert(data_buffer_mp0);
#if 0
  CopyoutSubgrid(elm_size_, num_dims_, data_[0], local_size_,
                 buf, sg_offset, sg_size);
#else
  CopyinoutSubgrid(
      1, elm_size_, num_dims_,
      data_buffer_mp0,
      local_size_, buf, sg_offset, sg_size
                   );
#endif
  buf += sg_size.accumulate(nd) * elm_size_;

  // copy diag
  // This should be safe
  sg_size[2] = halo_fw_width_[2];
  halo_size[2] = halo_fw_width_[2];
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_fw_[2],
                 halo_size, buf, sg_offset, sg_size);
  return;
}

// fw: prepare buffer for sending halo for forward access if true
void GridMPIOpenMP::CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal) {
  halo_has_diagonal_ = diagonal;

  // For the case dim == num_dims_ -1,
  // Change the below strategy so that we prepare
  // the actual tmpbuffer halo_self_{fw,bw}[2] and
  // copy the data_buffer_ to that tmpbuffer

  // So the following is changed:
#if 0
  if (dim == num_dims_ - 1) {
    if (fw) {
      halo_self_fw_[dim] = data_[0];
      return;
    } else {
      IntArray s = local_size_;
      s[dim] -= width;
      halo_self_bw_[dim] = data_[0]
          + s.accumulate(num_dims_) * elm_size_;
      return;
    }
  }
#endif

  char **halo_buf;
  SizeArray *halo_cur_size;
  if (fw) {
    halo_buf = &(halo_self_fw_[dim]);
    halo_cur_size = &halo_self_fw_buf_size_;
  } else {
    halo_buf = &(halo_self_bw_[dim]);
    halo_cur_size = &halo_self_bw_buf_size_;
  }

  // Ensure large enough buffer
  size_t halo_bytes = CalcHaloSize(dim, width, diagonal) * elm_size_;
  if ((*halo_cur_size)[dim] < halo_bytes) {
    (*halo_cur_size)[dim] = halo_bytes;
    *halo_buf = (char*)malloc(halo_bytes);
  }

#if 0  
  LOG_DEBUG() << "FW?: " << fw << ", width: " << width <<
      "local size: " << local_size_ << "\n";
#endif
  
  // If diagonal points are not required, copyoutSubgrid can be used
  // In the case dim == num_dims_ - 1, diagonal area is always not used
  if ((!diagonal) || (dim == num_dims_ - 1))
  {
    IntArray halo_offset;
    if (!fw) halo_offset[dim] = local_size_[dim] - width;
    IntArray halo_size = local_size_;
    halo_size[dim] = width;

    // And here uses data_[0]
#if 0
    CopyoutSubgrid(elm_size_, num_dims_, data_[0], local_size_,
                   *halo_buf, halo_offset, halo_size);
#else
    BufferHostOpenMP* data_buffer_mp0 = 
        dynamic_cast<BufferHostOpenMP*>(data_buffer_[0]);
    PSAssert(data_buffer_mp0);
    CopyinoutSubgrid(
        1, elm_size_, num_dims_,
        data_buffer_mp0,
        local_size_, *halo_buf, halo_offset, halo_size
                     );
#endif
    return;
  }

  switch (num_dims_) {
    case 2:
      CopyoutHalo2D0(width, fw, *halo_buf);
      break;
    case 3:
      if (dim == 0) {
        CopyoutHalo3D0(width, fw, *halo_buf);
      } else if (dim == 1) {
        CopyoutHalo3D1(width, fw, *halo_buf);
      } else {
        LOG_ERROR() << "This case should have been handled in the first block of this function.\n";
        PSAbort(1);
      }
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << num_dims_ << "\n";
      PSAbort(1);
  }
  
  return;
}

void GridMPIOpenMP::FixupBufferPointers() {
  BufferHostOpenMP *data_buffer_mp0 = 
      dynamic_cast<BufferHostOpenMP *>(data_buffer_[0]);
  BufferHostOpenMP *data_buffer_mp1 = 
      dynamic_cast<BufferHostOpenMP *>(data_buffer_[1]);
  PSAssert(data_buffer_mp0);
  PSAssert(data_buffer_mp1);

  data_MP_[0] = (char**)data_buffer_mp0->Get_MP();
  data_MP_[1] = (char**)data_buffer_mp1->Get_MP();
  data_[0] = 0;
  data_[1] = 0;
}


template <class T>
int ReduceGridMPIOpenMP(GridMPIOpenMP *g, PSReduceOp op, T *out) {
  size_t nelms = g->local_size().accumulate(g->num_dims());
  if (nelms == 0) return 0;
  boost::function<T (T, T)> func = GetReducer<T>(op);
#if 0
  T *d = (T *)g->_data();
  T v = d[0];
#else
  BufferHostOpenMP* data_buffer_mp0=
      dynamic_cast<BufferHostOpenMP*>(g->buffer());
  PSAssert(data_buffer_mp0);
  IntArray want_size(1, 1, 1);
  T v = (T) 0;
  IntArray emptyoffset;
  data_buffer_mp0->Copyout(
      &v, emptyoffset, want_size, 0);
#endif
  for (size_t i = 1; i < nelms; ++i) {
#if 0
    v = func(v, d[i]);
#else
    T buf = (T) 0;
    data_buffer_mp0->Copyout(
        &buf, emptyoffset, want_size, i);
    v = func(v, buf);
#endif
  }
  *out = v;
  return nelms;
}

int GridMPIOpenMP::Reduce(PSReduceOp op, void *out) {
  int rv = 0;
  switch (type_) {
    case PS_FLOAT:
      rv = ReduceGridMPIOpenMP<float>(this, op, (float*)out);
      break;
    case PS_DOUBLE:
      rv = ReduceGridMPIOpenMP<double>(this, op, (double*)out);
      break;
    default:
      PSAbort(1);
  }
  return rv;
}

void GridMPIOpenMP::InitNUMA(unsigned int maxMPthread)
{
#ifndef USE_OPENMP_NUMA
  return;
#else
#if 0
  BufferHostOpenMP* data_buffer_mp0 = 
      dynamic_cast<BufferHostOpenMP*>(data_buffer_[0]);
  PSAssert(data_buffer_mp0);
  data_buffer_mp0->InitNUMA(maxMPthread);
#else
  return;
#endif
#endif
}


GridSpaceMPIOpenMP::GridSpaceMPIOpenMP(
    int num_dims, const IntArray &global_size,
    int proc_num_dims, const IntArray &proc_size,
    int my_rank):
    GridSpaceMPI(num_dims, global_size,
                 proc_num_dims, proc_size,
                 my_rank)
{
}

GridSpaceMPIOpenMP::~GridSpaceMPIOpenMP() {
}


GridMPIOpenMP *GridSpaceMPIOpenMP::CreateGrid(PSType type, int elm_size,
                                              int num_dims,
                                              const IntArray &size,
                                              bool double_buffering,
                                              const IntArray &global_offset,
                                              const IntArray &division,
                                              int attr) {
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
  GridMPIOpenMP *g = GridMPIOpenMP::Create(
      type, elm_size, num_dims, grid_size,
      double_buffering,
      grid_global_offset, local_offset,
      local_size, division, attr);
  LOG_DEBUG() << "grid created\n";
  RegisterGrid(g);
  return g;
}

static void *ensure_buffer_capacity(void *buf, size_t cur_size,
                                    size_t required_size) {
  if (cur_size < required_size) {
    FREE(buf);
    cur_size = required_size;
    buf = malloc(cur_size);
  }
  return buf;
}

void GridSpaceMPIOpenMP::CollectPerProcSubgridInfo(
    const GridMPI *g,  const IntArray &grid_offset,
    const IntArray &grid_size,
    std::vector<FetchInfo> &finfo_holder) const {
  LOG_DEBUG() << "Collecting per-process subgrid info\n";
  IntArray grid_lim = grid_offset + grid_size;
  std::vector<FetchInfo> *fetch_info = new std::vector<FetchInfo>;
  std::vector<FetchInfo> *fetch_info_next = new std::vector<FetchInfo>;
  FetchInfo dummy;
  fetch_info->push_back(dummy);
  for (int d = 0; d < num_dims_; ++d) {
    // VERY CAREFUL !!
    const GridMPIOpenMP *g_dyn = dynamic_cast<const GridMPIOpenMP *>(g);
    PSAssert(g_dyn);

    ssize_t x = grid_offset[d] + g_dyn->global_offset_[d];
    for (int pidx = 0; pidx < proc_size_[d] && x < grid_lim[d]; ++pidx) {
      if (x < offsets_[d][pidx] + partitions_[d][pidx]) {
        LOG_VERBOSE_MPI() << "inclusion: " << pidx
                          << ", dim: " << d << "\n";
        FOREACH (it, fetch_info->begin(), fetch_info->end()) {
          FetchInfo info = *it;
          // peer process index
          info.peer_index[d] = pidx;
          // peer offset
          info.peer_offset[d] = x - g_dyn->global_offset_[d];
          // peer size
          info.peer_size[d] =
              std::min(offsets_[d][pidx] + partitions_[d][pidx] - x,
                       grid_lim[d] - x);
          fetch_info_next->push_back(info);
        }
        x = offsets_[d][pidx] + partitions_[d][pidx];
      }
    }
    std::swap(fetch_info, fetch_info_next);
    fetch_info_next->clear();
#if defined(PS_VERBOSE)
    FOREACH (it, fetch_info->begin(), fetch_info->end()) {
      const FetchInfo &finfo = *it;
      LOG_DEBUG() << "[" << my_rank_ << "] finfo peer index: "
                  << finfo.peer_index << "\n";
    }
#endif    
  }
  finfo_holder = *fetch_info;
  delete fetch_info;
  delete fetch_info_next;
  return;  
}

// Returns true if sent out
bool GridSpaceMPIOpenMP::SendFetchRequest(FetchInfo &finfo) const {
  int peer_rank = GetProcessRank(finfo.peer_index);
  size_t size = finfo.peer_size.accumulate(num_dims_);
  if (size == 0) return false; 
  SendGridRequest(my_rank_, peer_rank, comm_, FETCH_REQUEST);
  MPI_Request mpi_req;  
  CHECK_MPI(PS_MPI_Isend(&finfo, sizeof(FetchInfo), MPI_BYTE,
                         peer_rank, 0, comm_, &mpi_req));
  return true;
}

void GridSpaceMPIOpenMP::HandleFetchRequest(GridRequest &req, GridMPI *g) {
  LOG_DEBUG() << "HandleFetchRequest\n";
  FetchInfo finfo;
  int nd = num_dims_;
  CHECK_MPI(MPI_Recv(&finfo, sizeof(FetchInfo), MPI_BYTE,
                     req.my_rank, 0, comm_, MPI_STATUS_IGNORE));
  size_t bytes = finfo.peer_size.accumulate(nd) * g->elm_size();
  buf = ensure_buffer_capacity(buf, cur_buf_size, bytes);
  // char *_data() { return data_[0]; }
#if 0
  CopyoutSubgrid(g->elm_size(), nd, g->_data(), g->local_size(),
                 buf, finfo.peer_offset - g->local_offset(),
                 finfo.peer_size);
#else
  GridMPIOpenMP* g_MP = dynamic_cast<GridMPIOpenMP*>(g);
  PSAssert(g_MP);
  BufferHostOpenMP* g_data_buffer_MP =
      dynamic_cast<BufferHostOpenMP*>(g_MP->buffer());
  PSAssert(g_data_buffer_MP);
  g_MP->CopyinoutSubgrid(
      1, g_MP->elm_size(), nd,
      g_data_buffer_MP,
      g_MP->local_size(),
      buf, finfo.peer_offset - g->local_offset(),
      finfo.peer_size
                         );
#endif
  SendGridRequest(my_rank_, req.my_rank, comm_, FETCH_REPLY);
  MPI_Request mr;
  CHECK_MPI(PS_MPI_Isend(buf, bytes, MPI_BYTE, req.my_rank, 0, comm_, &mr));
  return;
}

void GridSpaceMPIOpenMP::HandleFetchReply(GridRequest &req, GridMPI *g,
                                          std::map<int, FetchInfo> &fetch_map,
                                          GridMPI *sg) {
  LOG_DEBUG() << "HandleFetchReply\n";
  const FetchInfo &finfo = fetch_map[req.my_rank];
  PSAssert(GetProcessRank(finfo.peer_index) == req.my_rank);
  size_t bytes = finfo.peer_size.accumulate(num_dims_) * g->elm_size();
  LOG_DEBUG() << "Fetch reply data size: " << bytes << "\n";
  buf = ensure_buffer_capacity(buf, cur_buf_size, bytes);
  CHECK_MPI(MPI_Recv(buf, bytes, MPI_BYTE, req.my_rank, 0,
                     comm_, MPI_STATUS_IGNORE));
  LOG_DEBUG() << "Fetch reply received\n";
#if 0
  CopyinSubgrid(sg->elm_size(), num_dims_, sg->_data(),
                sg->local_size(), buf,
                finfo.peer_offset - sg->local_offset(), finfo.peer_size);
#else
  GridMPIOpenMP* sg_MP = dynamic_cast<GridMPIOpenMP*>(g);
  PSAssert(sg_MP);
  BufferHostOpenMP* sg_data_buffer_MP =
      dynamic_cast<BufferHostOpenMP*>(sg_MP->buffer());
  PSAssert(sg_data_buffer_MP);
  sg_MP->CopyinoutSubgrid(
      false, sg_MP->elm_size(), num_dims_,
      sg_data_buffer_MP,
      sg_MP->local_size(),
      buf, finfo.peer_offset - sg->local_offset(),
      finfo.peer_size
                          );
#endif
  return;;
}

void *GridMPIOpenMP::GetAddress(const IntArray &indices_param) {
  IntArray indices = indices_param;
  // Use the remote grid if remote_grid_active is true.
  if (remote_grid_active()) {
    GridMPI *rmg = remote_grid();
    indices -= rmg->local_offset();

    // Need modification
#if 0
    return (void*)(rmg->_data() +
                   GridCalcOffset3D(indices, rmg->local_size())
                   * rmg->elm_size());
#else
    GridMPIOpenMP *rmg_MP = dynamic_cast<GridMPIOpenMP *>(rmg);
    PSAssert(rmg_MP);
    BufferHostOpenMP *buf_MP = 
        dynamic_cast<BufferHostOpenMP *>(rmg_MP->buffer());
    PSAssert(buf_MP);
    unsigned int cpuid = 0;
    size_t gridid = 0;
    size_t width_avail = 0;
    mpiopenmputil::getMPOffset(
        buf_MP->num_dims(), indices,
        buf_MP->size(), buf_MP->MPdivision(),
        buf_MP->MPoffset(),
        buf_MP->MPwidth(),
        cpuid, gridid, width_avail
                               );
    intptr_t pos = (intptr_t) ((buf_MP->Get_MP())[cpuid]);
    pos += gridid * buf_MP->elm_size();
    return (void *) pos;
  }
  
  indices -= local_offset();
  bool diag = halo_has_diagonal();
  for (int i = 0; i < num_dims(); ++i) {
    if (indices[i] < 0 || indices[i] >= local_size()[i]) {
      for (int j = i+1; j < PS_MAX_DIM; ++j) {
        if (diag) indices[i] += halo_bw_width()[i];
      }
      PSIndex offset;
      char *buf;
      if (indices[i] < 0) {
        indices[i] += halo_bw_width()[i];
        offset = GridCalcOffset3D(indices, halo_bw_size()[i]);
        buf = _halo_peer_bw()[i];
      } else {
        indices[i] -= local_size()[i];
        offset = GridCalcOffset3D(indices, halo_fw_size()[i]);
        buf = _halo_peer_fw()[i];
      }
      return (void*)(buf + offset*elm_size());
    }
  }

#if 0
  return (void*)(_data() +
                 GridCalcOffset3D(indices, local_size())
                 * elm_size());
#else
  {
    BufferHostOpenMP *buf_MP = 
        dynamic_cast<BufferHostOpenMP *>(buffer());
    PSAssert(buf_MP);
    unsigned int cpuid = 0;
    size_t gridid = 0;
    size_t width_avail = 0;
    mpiopenmputil::getMPOffset(
        buf_MP->num_dims(), indices,
        buf_MP->size(), buf_MP->MPdivision(),
        buf_MP->MPoffset(),
        buf_MP->MPwidth(),
        cpuid, gridid, width_avail
                               );
    intptr_t pos = (intptr_t) ((buf_MP->Get_MP())[cpuid]);
    pos += gridid *= buf_MP->elm_size();
    return (void *) pos;
  }
#endif
}
#endif

void GridMPIOpenMP::Copyin(void *dst, const void *src, size_t size){
  PSAssert(size <= (size_t) elm_size());
  Grid::Copyin(dst, src, size);
}

void GridMPIOpenMP::Copyout(void *dst, const void *src, size_t size){
  PSAssert(size <= (size_t) elm_size());
  Grid::Copyout(dst, src, size);
}


} // namespace runtime
} // namespace physis


