// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_GRID_H_
#define PHYSIS_RUNTIME_GRID_H_

#include <map>

#include "runtime/runtime_common.h"
#include "runtime/buffer.h"
#include "runtime/reduce.h"

namespace physis {
namespace runtime {

class Grid {
 protected:
  Grid(PSType type, int elm_size, int num_dims,
       const IndexArray &size,
       bool double_buffering, int attr):
      type_(type), elm_size_(elm_size), num_dims_(num_dims),
      size_(size),
      double_buffering_(double_buffering), attr_(attr) {
    num_elms_ = size.accumulate(num_dims_);
    data_[0] = NULL;
    data_[1] = NULL;
    data_buffer_[0] = NULL;
    data_buffer_[1] = NULL;
  }
  PSType type_;
 public:
  virtual ~Grid();

  static Grid* Create(PSType type, int elm_size, int num_dims,
                      const IndexArray &size,
                      bool double_buffering, int attr) {
    Grid *g = new Grid(type, elm_size, num_dims, size,
                       double_buffering, attr);
    g->InitBuffer();
    return g;
  }
  
  virtual std::ostream &Print(std::ostream &os) const;
  int &id() {
    return id_;
  }
  void Swap();
  PSType type() { return type_; }
  int elm_size_;
  int elm_size() const { return elm_size_; }
  int num_dims_;
  int num_dims() const { return num_dims_; }
  size_t num_elms_;
  virtual size_t num_elms() {return num_elms_; }
  IndexArray size_;
  const IndexArray &size() const { return size_; }
  bool double_buffering_;
  char *_data() { return data_[0]; }
  char *_data_emit() { return data_[1]; }  
  Buffer *buffer() { return data_buffer_[0]; }
  const Buffer *buffer() const { return data_buffer_[0]; }  
  virtual void *GetAddress(const IndexArray &indices);    
  virtual void Set(const IndexArray &indices, const void *buf);
  virtual void Get(const IndexArray &indices, void *buf); 
  bool AttributeSet(enum PS_GRID_ATTRIBUTE);

  //! Reduce the grid with operator op.
  /*
   * \param op The binary reduction operator.
   * \param out The buffer to store the reduced scalar value.
   * \return The number of reduced elements.
   */
  virtual int Reduce(PSReduceOp op, void *out);

  
  virtual void Save();
  virtual void Restore();
  
 protected:
  int id_;

  virtual void InitBuffer();
  virtual void DeleteBuffers();
  Buffer *data_buffer_[2];
  char *data_[2];
  int attr_;
  // the source is address of ordinary memory region
  virtual void Copyin(void *dst, const void *src, size_t size);
  // the dstination is address of ordinary memory region  
  virtual void Copyout(void *dst, const void *src, size_t size);
  virtual void SaveToFile(const void *buf, size_t len) const;
  virtual void RestoreFromFile(void *buf, size_t len);
  
};

class GridSpace {
 public:
  GridSpace() {}
  virtual ~GridSpace() {}
  Grid *FindGrid(int id) const;
  void DeleteGrid(Grid *g);
  void DeleteGrid(int id);
  //void ReduceGrid(Grid *g, void *buf);

  virtual void Save();
  virtual void Restore();
  
 protected:
  bool RegisterGrid(Grid *g);
  bool DeregisterGrid(Grid *g);
  std::map<int, Grid*> grids_;
  physis::Counter grid_counter_;  
};

} // namespace runtime
} // namespace physis

inline std::ostream& operator<<(std::ostream &os, physis::runtime::Grid &g) {
  return g.Print(os);
}


#endif /* PHYSIS_RUNTIME_GRID_H_ */

