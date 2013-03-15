// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/buffer.h"

namespace physis {
namespace runtime {

Buffer::Buffer(BufferDeleter deleter): size_(0), buf_(NULL), deleter_(deleter) {
}

Buffer::~Buffer() {
  Free();
}

void Buffer::Allocate(size_t size) {
  PSAssert(buf_ == NULL);
  buf_ = GetChunk(size);
  size_ = size;
}

void Buffer::EnsureCapacity(size_t size) {
  LOG_VERBOSE() << "Current size: " << size_ << "; requested size: " <<
      size << "\n";
  if (size >= size_) {
    LOG_INFO() << "Expanding capacity from " << size_
                  << " to " << size << "\n";
    Free();
    Allocate(size);
  }
}

void Buffer::Shrink(size_t size) {
  if (!(size <= size_)) {
    LOG_VERBOSE() << "Shrinking buffer from " << size_
                  << " bytes to " << size << "bytes.\n";
    Free();
    Allocate(size);
  }
}

void Buffer::Free() {
  if (deleter_) deleter_(buf_);
  buf_ = NULL;
  size_ = 0;
}

//
// BufferHost
//

BufferHost::BufferHost(): Buffer(free) {
}

BufferHost::~BufferHost() {}

void *BufferHost::GetChunk(size_t size) {
  if (size == 0) return NULL;
  void *p = calloc(1, size);
  PSAssert(p);
  return p;
}


} // namespace runtime
} // namespace physis
