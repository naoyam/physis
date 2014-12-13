// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/buffer.h"
#include "runtime/grid_util.h"

namespace physis {
namespace runtime {

Buffer::Buffer(BufferDeleter deleter): size_(0), actual_size_(0), buf_(NULL), deleter_(deleter) {
}

Buffer::~Buffer() {
  Free();
}

void Buffer::Allocate(size_t size) {
  PSAssert(buf_ == NULL);
  buf_ = GetChunk(size);
  actual_size_ = size_ = size;
}

void Buffer::EnsureCapacity(size_t size) {
  LOG_VERBOSE() << "Current logical size: " << size_
                << "; current actual size: " << actual_size_
                << "; requested size: " << size << "\n";
  if (size >= actual_size_) {
    LOG_INFO() << "Expanding capacity from " << actual_size_
                  << " to " << size << "\n";
    Free();
    Allocate(size);
  } else if (size != size_) {
    LOG_DEBUG() << "Changing logical size from " << size_ << " to " << size << "\n";
    size_ = size;
  }
}

// void Buffer::Shrink(size_t size) {
//   if (!(size <= actual_size_)) {
//     LOG_VERBOSE() << "Shrinking buffer from " << actual_size_
//                   << " bytes to " << size << "bytes.\n";
//     Free();
//     Allocate(size);
//   }
// }

void Buffer::Free() {
  if (deleter_) deleter_(buf_);
  buf_ = NULL;
  actual_size_ = size_ = 0;
}

//
// BufferHost
//

BufferHost::BufferHost(): Buffer(free) {
}

BufferHost::BufferHost(BufferDeleter deleter): Buffer(deleter) {
}

BufferHost::~BufferHost() {}

void *BufferHost::GetChunk(size_t size) {
  if (size == 0) return NULL;
  void *p = calloc(1, size);
  PSAssert(p);
  return p;
}

void BufferHost::Copyout(void *dst, size_t size) {
  memcpy(dst, Get(), size);
}

void BufferHost::Copyout(size_t elm_size, int rank,
                         const IndexArray  &grid_size,
                         void *subgrid,
                         const IndexArray &subgrid_offset,
                         const IndexArray &subgrid_size) {

  CopyoutSubgrid(elm_size, rank, Get(),
                 grid_size, subgrid, subgrid_offset, subgrid_size);
}

void BufferHost::Copyin(const void *src, size_t size) {
  memcpy(Get(), src, size);
}

void BufferHost::Copyin(size_t elm_size, int rank,
                        const IndexArray  &grid_size,
                        const void *subgrid,
                        const IndexArray &subgrid_offset,
                        const IndexArray &subgrid_size) {

  CopyinSubgrid(elm_size, rank, Get(),
                grid_size, subgrid, subgrid_offset, subgrid_size);
}

} // namespace runtime
} // namespace physis
