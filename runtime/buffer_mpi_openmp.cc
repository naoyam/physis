#include "runtime/buffer_mpi_openmp.h"
#undef USE_LINESIZE

#ifdef USE_OPENMP_NUMA
#include <numaif.h>
#include <numa.h>
#endif

namespace physis {
namespace runtime {

void BufferHostOpenMP::Allocate(
    int num_dims, size_t elm_size, const IntArray &size){

  DeleteOpenMP();

  IntArray unitarray(1,1,1);
  IntArray requested_size = size;
  // The following MUST be commented out
  //requested_size.SetNoLessThan(unitarray);

  if (size.accumulate(num_dims)) {
    num_dims_ = num_dims;
    elm_size_ = elm_size;
  }
  size_  = size;
  const size_t alignment = sysconf(_SC_PAGESIZE);

  int status = CreateAlignedMultiBuffer(
      requested_size, division_,
      elm_size,
      alignment,
      &buf_mp_,
      mp_offset_, mp_width_,
      mp_cpu_memsize_, mp_cpu_allocBytes_,
      mp_linesize_
                                        );

  if (status) {
    LOG_ERROR() << "Calling CreateAlignedMultiBuffer failed\n";
    PSAbort(1);
  }

#ifdef USE_OPENMP_NUMA
  int maxcpunodes =  0;
  if (numa_available() != -1){
    maxcpunodes = numa_max_node() + 1;
    LOG_DEBUG() << "NUMA reported that " << maxcpunodes << " nodes may be available\n";
    MoveMultiBuffer(maxcpunodes);
  }
#endif
  
}

void BufferHostOpenMP::Allocate(const IntArray &size) {
  Allocate(num_dims_, elm_size_, size);
}


//
// BufferHost
//

BufferHostOpenMP::BufferHostOpenMP(size_t elm_size):
    BufferHost(elm_size),
    buf_mp_(0),
    mp_offset_(0),
    mp_width_(0),
    mp_linesize_(1),
    mp_cpu_memsize_(0),
    mp_cpu_allocBytes_(0),
    mbind_done_p(false)
{
}
BufferHostOpenMP::BufferHostOpenMP(int num_dims,  size_t elm_size):
    BufferHost(num_dims, elm_size),
    buf_mp_(0),
    mp_offset_(0),
    mp_width_(0),
    mp_linesize_(1),
    mp_cpu_memsize_(0),
    mp_cpu_allocBytes_(0),
    mbind_done_p(false)
{
  IntArray minimum(1,1,1);
  division_ = minimum;
}

BufferHostOpenMP::BufferHostOpenMP(int num_dims,  size_t elm_size, IntArray &division_in):
    BufferHost(num_dims, elm_size),
    buf_mp_(0),
    division_(division_in),
    mp_offset_(0),
    mp_width_(0),
    mp_linesize_(1),
    mp_cpu_memsize_(0),
    mp_cpu_allocBytes_(0),
    mbind_done_p(false)
{
  IntArray unitarray(1,1,1);
  division_.SetNoLessThan(unitarray);
}

BufferHostOpenMP::~BufferHostOpenMP() {
  DeleteOpenMP();
}

void BufferHostOpenMP::Copyin(
    const void *buf, const IntArray &offset,
    const IntArray &size){

  Copyin(buf, offset, size, (size_t) 0);

}

void BufferHostOpenMP::Copyin(
    const void *buf, const IntArray &offset,
    const IntArray &size,
    const size_t linear_offset
                              ){
  EnsureCapacity(offset + size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);

  size_t size_left = GetLinearSize(size);
  size_t offset_left = linear_offset * elm_size_;

  intptr_t buf_curpos = (intptr_t)buf;

  unsigned int cpucount[PS_MAX_DIM] = {0};
  unsigned int xyzcount[PS_MAX_DIM] = {0};
  for (cpucount[2] = 0; cpucount[2] < (unsigned)division_[2] ; cpucount[2]++) {
    for (xyzcount[2] = 0; xyzcount[2] <  mp_width_[2][cpucount[2]]; xyzcount[2]++) {

      for (cpucount[1] = 0; cpucount[1] < (unsigned)division_[1] ; cpucount[1]++) {
        for (xyzcount[1] = 0; xyzcount[1] <  mp_width_[1][cpucount[1]]; xyzcount[1]++) {

          for (cpucount[0] = 0; cpucount[0] < (unsigned)division_[0] ; cpucount[0]++) {
            //for (xyzcount[0] = 0; xyzcount[0] <  mp_width_[0][cpucount[0]]; xyzcount[0]++) {

            unsigned int cpuid =
                cpucount[0] + 
                cpucount[1] * division_[0] +
                cpucount[2] * division_[0] * division_[1];
            unsigned long int offset =
#ifdef USE_LINESIZE
                xyzcount[1] * mp_linesize_ +
                xyzcount[2] * mp_linesize_ * mp_width_[1][cpucount[1]];
#else
            xyzcount[1] * mp_width_[0][cpucount[0]] +
                xyzcount[2] * mp_width_[0][cpucount[0]] * mp_width_[1][cpucount[1]];
#endif
            unsigned long int size_writenow = mp_width_[0][cpucount[0]];
            size_writenow *= elm_size_;

            intptr_t mp_curpos = ((intptr_t)Get_MP()[cpuid] + offset * elm_size_);
            if (offset_left) {
              if (size_writenow > offset_left) {
                size_writenow -= offset_left;
                mp_curpos += offset_left;
                offset_left = 0;
              } else {
                offset_left -= size_writenow;
                size_writenow = 0;
              } 
            }

            if (size_writenow > size_left) size_writenow = size_left;
            if (size_writenow) {
              memcpy((void *)mp_curpos, (const void *)buf_curpos, size_writenow);
              size_left -= size_writenow;
              buf_curpos += size_writenow;
            }

            if (!size_left)
              return;

            //} // xyzcount[0]
          } // cpucount[0]
        } // xyzcount[1]
      } // cpucount[1]
    } // xyzcount[2]
  } // cpucount[2]

  LOG_ERROR() << "Should not reach here!!\n";
  PSAbort(1);

}

void BufferHostOpenMP::Copyout(
    void *buf, const IntArray &offset,
    const IntArray &size){

  Copyout(buf, offset, size, (size_t) 0);

}

void BufferHostOpenMP::Copyout(
    void *buf, const IntArray &offset,
    const IntArray &size,
    const size_t linear_offset)
{
  EnsureCapacity(offset + size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);

  size_t size_left = GetLinearSize(size);
  size_t offset_left = linear_offset * elm_size_;

  intptr_t buf_curpos = (intptr_t)buf;

  unsigned int cpucount[PS_MAX_DIM] = {0};
  unsigned int xyzcount[PS_MAX_DIM] = {0};
  for (cpucount[2] = 0; cpucount[2] < (unsigned)division_[2] ; cpucount[2]++) {
    for (xyzcount[2] = 0; xyzcount[2] <  mp_width_[2][cpucount[2]]; xyzcount[2]++) {

      for (cpucount[1] = 0; cpucount[1] < (unsigned)division_[1] ; cpucount[1]++) {
        for (xyzcount[1] = 0; xyzcount[1] <  mp_width_[1][cpucount[1]]; xyzcount[1]++) {

          for (cpucount[0] = 0; cpucount[0] < (unsigned)division_[0] ; cpucount[0]++) {
            //for (xyzcount[0] = 0; xyzcount[0] <  mp_width_[0][cpucount[0]]; xyzcount[0]++) {

            unsigned int cpuid =
                cpucount[0] + 
                cpucount[1] * division_[0] +
                cpucount[2] * division_[0] * division_[1];
            unsigned long int offset =
#ifdef USE_LINESIZE
                xyzcount[1] * mp_linesize_ +
                xyzcount[2] * mp_linesize_ * mp_width_[1][cpucount[1]];
#else
            xyzcount[1] * mp_width_[0][cpucount[0]] +
                xyzcount[2] * mp_width_[0][cpucount[0]] * mp_width_[1][cpucount[1]];
#endif
            unsigned long int size_writenow = mp_width_[0][cpucount[0]];
            size_writenow *= elm_size_;

            intptr_t mp_curpos = ((intptr_t)Get_MP()[cpuid] + offset * elm_size_);
            if (offset_left) {
              if (size_writenow > offset_left) {
                size_writenow -= offset_left;
                mp_curpos += offset_left;
                offset_left = 0;
              } else {
                offset_left -= size_writenow;
                size_writenow = 0;
              } 
            }

            if (size_writenow > size_left) size_writenow = size_left;
            if (size_writenow) {
              memcpy((void *)buf_curpos, (const void *)mp_curpos, size_writenow);
              size_left -= size_writenow;
              buf_curpos += size_writenow;
            }

            if (!size_left)
              return;

            //} // xyzcount[0]
          } // cpucount[0]
        } // xyzcount[1]
      } // cpucount[1]
    } // xyzcount[2]
  } // cpucount[2]

  LOG_ERROR() << "Should not reach here!!\n";
  PSAbort(1);

}



void BufferHostOpenMP::MPISendRecvInoI(
    int recv_p,
    int blocking_p,
    int srcdstNUM,
    MPI_Comm comm, MPI_Request *req,
    const IntArray &offset, const IntArray &size,
    const size_t *cpu_memsize
                                       ){
  // offset access is not yet support
  PSAssert(offset == 0);
  if (recv_p) {
    EnsureCapacity(offset + size);
  } else {
    PSAssert(offset + size < size_);
  }

  size_t size_left = GetLinearSize(size);
  int cpucount;
  for (cpucount = 0; cpu_memsize[cpucount]; cpucount++) {
    size_t size_now = cpu_memsize[cpucount];
    size_now *= elm_size_;
    if (size_now > size_left) size_now = size_left;

    void *buf = (Get_MP())[cpucount];

    if (recv_p) {
      if (blocking_p) {
        PS_MPI_Recv(buf, size_now, MPI_BYTE, srcdstNUM, 0,
                    comm, MPI_STATUS_IGNORE);
      }
      if (!blocking_p) {
      }
      PS_MPI_Irecv(buf, size_now, MPI_BYTE, srcdstNUM, 0, comm, req);
    }
    if (!recv_p) {
      if (blocking_p) {
        PS_MPI_Send(buf, size_now, MPI_BYTE, srcdstNUM, 0, comm);  
      }
      if (!blocking_p) {
        PS_MPI_Isend(buf, size_now, MPI_BYTE, srcdstNUM, 0, comm, req);  
      }
    }

    size_left -= size_now;
    if (!size_left) return;
  } // for (cpucount = 0; cpu_memsize[cpucount]; cpucount++)

  LOG_ERROR() << "Should not reach here!!\n";
  PSAbort(1);
}

// For example
// num_dims: 3 <dimension>
// num_procs: 6 = 1*1*6
// size = global_size: {64, 64, 64}
// num_partitions = proc_size: {1, 1, 6}
// {{64}, {64}, {10,10,11,11,11,11}}

int BufferHostOpenMP::CreateAlignedMultiBuffer(
    const IntArray &requested_size, IntArray &division,
    const int elmsize,
    const size_t alignment,
    void ***ret_buf,
    size_t **&ret_offset, size_t **&ret_width,
    size_t *&ret_cpu_memsize,
    size_t *&ret_cpu_allocbytes,
    size_t &linesize
                                               ){

  int status = 0;

  LOG_DEBUG() << "Allocate size:" << requested_size << ".\n";

  IntArray req_expanded_size = requested_size;

  IntArray unitarray(1,1,1);
  req_expanded_size.SetNoLessThan(unitarray);
  division.SetNoMoreThan(req_expanded_size);

  ret_offset = new size_t*[PS_MAX_DIM];
  ret_width = new size_t*[PS_MAX_DIM];

  for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++) {
    ret_offset[dim] = new size_t[division[dim]];
    ret_width[dim] = new size_t[division[dim]];
    for (unsigned int j = 0; j < (unsigned)division[dim]; j++) {
      size_t minc = requested_size[dim] * j / division[dim];
      size_t maxc = requested_size[dim] * (j + 1) / division[dim];
      if (j == (unsigned)division[dim] - 1)
        maxc = requested_size[dim];

      ret_offset[dim][j] = minc;
      ret_width[dim][j] = maxc - minc;

      if (dim == 0) {
        if (linesize < ret_width[dim][j])
          linesize = ret_width[dim][j];
      }

    }
  }

  unsigned int total_num = 1;
  for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++)
    total_num *= division[dim];

  *ret_buf = (void **) calloc(total_num, sizeof(void *));
  ret_cpu_memsize = new size_t[total_num + 1];
  ret_cpu_memsize[total_num] = 0;
  ret_cpu_allocbytes = new size_t[total_num + 1];
  ret_cpu_allocbytes[total_num] = 0;
  
  if (! *ret_buf) {
    LOG_ERROR() << "Allocating buffer failed.\n";
    return 1;
  }

  for (unsigned int cpunum = 0; cpunum < total_num; cpunum++) {
    size_t real_size = 1;
    size_t unit = 1;
    for (int dim = 0; dim < PS_MAX_DIM; dim++) {
      unsigned int id = (cpunum / unit) % division[dim];
      unit *= division[dim];
      size_t width_tmp = ret_width[dim][id];
#ifdef USE_LINESIZE
      if (dim == 0) width_tmp = linesize;
#endif
      real_size *= width_tmp;
    }

    ret_cpu_memsize[cpunum] = real_size;
    real_size *= elmsize;

    size_t allocate_size =
        ((real_size + alignment - 1) / alignment) * alignment;
    ret_cpu_allocbytes[cpunum] = allocate_size;
    LOG_DEBUG() << real_size << " bytes requested, actually allocating "
                << allocate_size << " bytes "
                << "to cpuid " << cpunum << "\n";

    void *ptr = 0;
    if (real_size) {
      status = posix_memalign(&ptr, alignment, allocate_size);
      if (status) {
        LOG_ERROR() << "posix_memalign failed with status "
                    << status << "\n";
        return 1;
      }
    }
    (*ret_buf)[cpunum] = ptr;
  }

  return status;

}


void BufferHostOpenMP::DestroyMultiBuffer(
    void ***src_buf, const IntArray &division
                                          ){

  unsigned int total_num = 1;
  for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++)
    total_num *= division[dim];

  void **arr_buf = *src_buf;
  if (!arr_buf) return;

  for (unsigned int num = 0; num < total_num; num++) {
    if (arr_buf[num]) free(arr_buf[num]);
  }
  free(arr_buf);
  arr_buf = 0;
}

void BufferHostOpenMP::DestroyMP3Dinfo(
    size_t **&src_3dinfo, const IntArray &division
                                       ){
  if (! src_3dinfo) return;

  for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++) {
    if (src_3dinfo[dim]) delete[] src_3dinfo[dim];
  }

  delete [] src_3dinfo;
  src_3dinfo = 0;
}


void BufferHostOpenMP::DeleteOpenMP()
{
  DestroyMultiBuffer(&buf_mp_, division_);
  DestroyMP3Dinfo(mp_offset_, division_);
  DestroyMP3Dinfo(mp_width_, division_);

  if (mp_cpu_memsize_) delete mp_cpu_memsize_;
  if (mp_cpu_allocBytes_) delete mp_cpu_allocBytes_;
  mp_cpu_memsize_ = 0;
  mp_cpu_allocBytes_ = 0;
  
}

} // namespace runtime
} // namespace physis
