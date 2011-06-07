// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/buffer.h"

namespace physis {
namespace runtime {

Buffer::Buffer(size_t elm_size):
    num_dims_(1), elm_size_(elm_size), size_((index_t)0), buf_(NULL),
    deleter_(NULL) {
}

Buffer::Buffer(int num_dims, size_t elm_size):
    num_dims_(num_dims), elm_size_(elm_size), size_((index_t)0), buf_(NULL),
    deleter_(NULL) {
}

Buffer::~Buffer() {
  Delete();
}

void Buffer::Copyin(const void *buf, size_t size) {
  IntArray s;
  s.assign(1);
  s[0] = size;
  IntArray offset((index_t)0);
  Copyin(buf, offset, s);
}

void *Buffer::Copyout() {
  size_t s = GetLinearSize();
  void *buf = malloc(s);
  Copyout(buf, IntArray((index_t)0), size_);
  return buf;
}

void Buffer::EnsureCapacity(size_t elm_size, size_t size) {
  EnsureCapacity(1, elm_size, IntArray(size));
}

void Buffer::EnsureCapacity(int num_dims, size_t elm_size,
                            const IntArray &size) {
  size_t cur_size = GetLinearSize();
  size_t target_size = Buffer::GetLinearSize(num_dims, elm_size, size);
  LOG_VERBOSE() << "Current size: " << cur_size << "; requested size: " <<
    target_size << "\n";
  if (!(cur_size >= target_size)) {
    LOG_INFO() << "Expanding capacity from " << size_
                  << " to " << size << "\n";
    Allocate(num_dims, elm_size, size);
  }
}

void Buffer::Shrink(size_t size) {
  Shrink(IntArray(size));
}

void Buffer::Shrink(const IntArray &size) {  
  if (!(size <= size_)) {
    LOG_VERBOSE() << "Shrinking buffer from " << size_
                  << " bytes to " << size << "bytes.\n";
    Allocate(size);
  }
}

void Buffer::Delete() {
  if (buf_) {
    DeleteChunk(buf_);
  }
  buf_ = NULL;
  size_.assign(0);
}

void Buffer::Allocate(int num_dims, size_t elm_size, const IntArray &size) {
  Delete();
  if (size.accumulate(num_dims)) {
    num_dims_ = num_dims;
    elm_size_ = elm_size;
    buf_ = GetChunk(size);
    if (!buf_) {
      LOG_ERROR() << "Buffer allocation failure\n";
      PSAbort(1);
    }
  }
  size_ = size;
}

void Buffer::Allocate(const IntArray &size) {
  Allocate(num_dims_, elm_size_, size);
}


void *Buffer::GetChunk(const IntArray &size) {
  LOG_ERROR() << "This is a dummy implementation, and should be overridden by child classes.\n";
  PSAbort(1);
  return NULL;
}

void Buffer::DeleteChunk(void *p) {
  PSAssert(deleter_);
  deleter_(p);
}

//
// BufferHost
//

BufferHost::BufferHost(size_t elm_size):
    Buffer(elm_size) {
  deleter_ = free;
}
BufferHost::BufferHost(int num_dims,  size_t elm_size):
    Buffer(num_dims, elm_size) {
  deleter_ = free;
}

BufferHost::~BufferHost() {
}

void BufferHost::Copyin(const void *buf, const IntArray &offset,
                        const IntArray &size) {
  EnsureCapacity(offset + size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(Get(), buf, GetLinearSize(size));
}

void BufferHost::Copyout(void *buf, const IntArray &offset,
                         const IntArray &s) {
  PSAssert(offset + s <= size());
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(buf, Get(), GetLinearSize(s));
}

void BufferHost::MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                         const IntArray &size) {
  EnsureCapacity(offset + size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);
#ifdef PS_DEBUG  
  PSAssert(GetLinearSize() >= GetLinearSize(size));
#endif
  PS_MPI_Recv(Get(), GetLinearSize(size), MPI_BYTE, src, 0, comm,
              MPI_STATUS_IGNORE);
}

void BufferHost::MPIIrecv(int src, MPI_Comm comm, MPI_Request *req,
                          const IntArray &offset, const IntArray &size) {
  EnsureCapacity(offset + size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);
#ifdef PS_DEBUG  
  PSAssert(GetLinearSize() >= GetLinearSize(size));
#endif
  
  PS_MPI_Irecv(Get(), GetLinearSize(size), MPI_BYTE, src, 0, comm,
               req);
  
}

void BufferHost::MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                         const IntArray &s) {
  PSAssert(offset + s <= size());
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  PS_MPI_Send(Get(), GetLinearSize(s), MPI_BYTE, dst, 0, comm);  
}

void BufferHost::MPIIsend(int dst, MPI_Comm comm, MPI_Request *req,
                          const IntArray &offset, const IntArray &s) {
  PSAssert(offset + s <= size());
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  PS_MPI_Isend(Get(), GetLinearSize(s), MPI_BYTE, dst, 0, comm, req);  
}

  
void *BufferHost::GetChunk(const IntArray &size) {
  size_t s = size.accumulate(num_dims_);
  if (s == 0) return NULL;
  return calloc(elm_size_, s);
}

// void BufferHost::DeleteChunk(void *p) {
//   free(p);
// }

void BufferHost::Copyin(const BufferHost &buf, const IntArray &offset,
                        const IntArray &size) {
  PSAssert(buf.num_dims() == 1);
  Copyin(buf.Get(), offset, size);
}

void BufferHost::Copyout(BufferHost &buf,  const IntArray &offset,
                         const IntArray &size) {
  PSAssert(buf.num_dims() == 1);
  Copyout(buf.Get(), offset, size);
}

} // namespace runtime
} // namespace physis
