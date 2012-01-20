#include "runtime/buffer_opencl.h"
#include "runtime/rpc_opencl_common.h"

#include <CL/cl.h>

#define OpenCL_MEMCPY_ASYNC_SIZE (64 << 10) // 64KB

namespace physis {
namespace runtime {

BufferOpenCLHost::BufferOpenCLHost(size_t elm_size)
    : Buffer(elm_size) {
  mpi_buf_ = new BufferHost(elm_size);
  deleter_ = BufferOpenCLHost::DeleteChunk;
}

BufferOpenCLHost::BufferOpenCLHost(int num_dims,  size_t elm_size)
    : Buffer(num_dims, elm_size) {
  mpi_buf_ = new BufferHost(num_dims, elm_size);
  deleter_ = BufferOpenCLHost::DeleteChunk;  
}

BufferOpenCLHost::~BufferOpenCLHost() {
  delete mpi_buf_;
}

void BufferOpenCLHost::DeleteChunk(void *ptr) {
  free(ptr);
  return;
}

void *BufferOpenCLHost::GetChunk(const IntArray &size) {
  void *ptr = NULL;
  if (size.accumulate(num_dims_) > 0) {
    LOG_INFO() << "Trying to allocate host pinned memory of "
	       << GetLinearSize(size) << " bytes.\n";
    ptr = (void *) malloc(GetLinearSize(size));
  }
  return ptr;
}

void BufferOpenCLHost::Copyin(const void *buf, const IntArray &offset,
                            const IntArray &size) {
  EnsureCapacity(offset+size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(Get(), buf, GetLinearSize(size));
}

void BufferOpenCLHost::Copyin(const BufferHost &buf, const IntArray &offset,
                            const IntArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferOpenCLHost::Copyout(void *buf, const IntArray &offset,
                             const IntArray &s) {
  PSAssert(offset + s <= size());
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(buf, Get(), GetLinearSize(s));
}

void BufferOpenCLHost::Copyout(BufferHost &buf,
                             const IntArray &offset,
                             const IntArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  Copyout(buf.Get(), offset, size);
}


void BufferOpenCLHost::MPIRecv(int src, MPI_Comm comm,
                             const IntArray &offset,
                             const IntArray &size) {
  mpi_buf_->MPIRecv(src, comm, IntArray((index_t)0), size);
  Copyin(*mpi_buf_, offset, size);
  //mpi_buf_->Delete();
}

void BufferOpenCLHost::MPISend(int dst, MPI_Comm comm,
                             const IntArray &offset,
                             const IntArray &size) {
  Copyout(*mpi_buf_, offset, size);
  mpi_buf_->MPISend(dst, comm, IntArray((index_t)0), size);
  //mpi_buf_->Delete();
}

//
// BufferOpenCLDev
//
BufferOpenCLDev::BufferOpenCLDev(size_t elm_size, CLbaseinfo *clinfo_in)
    : Buffer(elm_size),
      base_clinfo_(clinfo_in), stream_clinfo_(0) /*, strm_(0)*/,
      buf_mem(0)
{
  pinned_buf_ = new BufferOpenCLHost(elm_size);
  deleter_ = BufferOpenCLDev::DeleteChunk; // But this deleter_ won't be used
  pitch = 0;
}

BufferOpenCLDev::BufferOpenCLDev(int num_dims, size_t elm_size, CLbaseinfo *clinfo_in)
    : Buffer(num_dims, elm_size),
      base_clinfo_(clinfo_in), stream_clinfo_(0) /*, strm_(0)*/,
      buf_mem(0)
{
  pinned_buf_ = new BufferOpenCLHost(num_dims, elm_size);
  deleter_ = BufferOpenCLDev::DeleteChunk;  // But this deleter_ won't be used
  pitch = 0;
}

BufferOpenCLDev::~BufferOpenCLDev() {
  delete pinned_buf_;  
}


const void *BufferOpenCLDev::Get() const 
{
  LOG_DEBUG() << "Never use BufferOpenCLDev::Get()!!!\n";
  PSAbort(1);

  return 0;
}

void *&BufferOpenCLDev::Get() {
  LOG_DEBUG() << "Never use BufferOpenCLDev::Get()!!!\n";
  PSAbort(1);

  return buf_;
}

void BufferOpenCLDev::Copyin(const void *buf, const IntArray &offset,
                           const IntArray &size) {
  pinned_buf_->Copyin(buf, size);
  Copyin(*pinned_buf_, offset, size);  
}

void BufferOpenCLDev::Copyin(const BufferHost &buf, const IntArray &offset,
                           const IntArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferOpenCLDev::Copyin(const BufferOpenCLHost &buf,
                           const IntArray &offset,
                           const IntArray &size) {
  PSAssert(offset == 0);
  EnsureCapacity(offset+size);

  // CUDA code waits here until stream completes all operations
  CLbaseinfo *clinfo_use = stream_clinfo();
  if (!clinfo_use) {
    LOG_DEBUG() << "Currently stream not set\n";
    clinfo_use = base_clinfo();
  }
  PSAssert(clinfo_use);
  cl_command_queue buf_queue_ = clinfo_use->get_queue();
  PSAssert(buf_queue_ != 0);
  cl_int status = clEnqueueWriteBuffer(
      buf_queue_, Get_buf_mem(), CL_FALSE, /* Once no block */
      0, GetLinearSize(size), buf.Get(),
      0, NULL, NULL);
  if (status != CL_SUCCESS) {
    LOG_DEBUG() << "Calling clEnqueueWriteBuffer() failed.\n";
  }
  // And block
  clFinish(buf_queue_);
}

void BufferOpenCLDev::Copyout(void *buf, const IntArray &offset,
                            const IntArray &size) {
  Copyout(*pinned_buf_, offset, size);
  pinned_buf_->Copyout(buf, size);
}

void BufferOpenCLDev::Copyout(BufferHost &buf, const IntArray &offset,
                            const IntArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferOpenCLDev::Copyout(BufferOpenCLHost &buf, const IntArray &offset,
                            const IntArray &size) {
  PSAssert(offset == 0);
  PSAssert(offset + size <= this->size());
  buf.EnsureCapacity(num_dims_, elm_size_, size);

  // CUDA code waits here until stream completes all operations
  CLbaseinfo *clinfo_use = stream_clinfo();
  if (!clinfo_use) {
    LOG_DEBUG() << "Currently stream not set\n";
    clinfo_use = base_clinfo();
  }
  PSAssert(clinfo_use);
  cl_command_queue buf_queue_ = clinfo_use->get_queue();
  PSAssert(buf_queue_ != 0);
  cl_int status = clEnqueueReadBuffer(
      buf_queue_, Get_buf_mem(), CL_FALSE, /* Once no block */
      0, GetLinearSize(size), buf.Get(),
      0, NULL, NULL);
  if (status != CL_SUCCESS) {
    LOG_DEBUG() << "Calling clEnqueueReadBuffer() failed.\n";
  }
  // And block
  clFinish(buf_queue_);
}

  
void BufferOpenCLDev::MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                            const IntArray &size) {
  // First, recv with the host pinned buffer (which also performs
  // internal copying between MPI and OpenCL buffers.
  pinned_buf_->Buffer::MPIRecv(src, comm, size);
  // Then use CLEnqueueWriteBuffer to copy into the device memory
  Copyin(*pinned_buf_, offset, size);
}

void BufferOpenCLDev::MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                            const IntArray &size) {
  Copyout(*pinned_buf_, offset, size);
  pinned_buf_->Buffer::MPISend(dst, comm, size);
}

void BufferOpenCLDev::GetChunk_CL(
  const IntArray &size, cl_mem *ret_p_mem, 
  size_t *ret_pitch
)
{
  *ret_p_mem = 0;
  // FIXME
  // FIXME
  // NEEDCHECK
  *ret_pitch = size[0] * elm_size_;
  if (size.accumulate(num_dims_) >0) {
    cl_int status;
    CLbaseinfo *clinfo_use = stream_clinfo();
    if (!clinfo_use)
      clinfo_use = base_clinfo();
    PSAssert(clinfo_use);

    cl_context buf_context_ = clinfo_use->get_context();
    PSAssert(buf_context_);
    *ret_p_mem = clCreateBuffer(
      buf_context_, CL_MEM_READ_WRITE, GetLinearSize(size), NULL, &status);
    if (status != CL_SUCCESS)
      LOG_DEBUG() << "Calling clCreateBuffer failed\n";
  }
}

void *BufferOpenCLDev::GetChunk(const IntArray &size) {
  LOG_ERROR() << "Never use BufferOpenCLDev::GetChunk!!!\n";
  PSAbort(1);
  return 0;
}

void BufferOpenCLDev::DeleteChunk_CL(cl_mem p_mem) {
  if (p_mem) {
    clReleaseMemObject(p_mem);
  }
  p_mem = 0;
}

void BufferOpenCLDev::DeleteChunk(void *ptr) {
  LOG_ERROR() << "Never use BufferOpenCLDev::DeleteChunk!!!\n";
  PSAbort(1);
}

void BufferOpenCLDev::Allocate(int num_dims, size_t elm_size, const IntArray &size){
  Delete();
  if (size.accumulate(num_dims)) {
    num_dims_ = num_dims;
    elm_size_ = elm_size;
    GetChunk_CL(size, &buf_mem, &pitch);
    if (!buf_mem) {
      LOG_ERROR() << "Buffer allocation failure\n";
      PSAbort(1);
    }
  }
  size_ = size;
}

void BufferOpenCLDev::Delete() {
  if (buf_mem) {
    DeleteChunk_CL(buf_mem);
  }
  buf_mem = 0;
  size_.assign(0);
}

//
// BufferOpenCLDev3D
// 
 
#if 0 
#endif
  

} // namespace runtime
} // namespace physis
