// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/grid.h"

#include "physis/physis_util.h"
#include "runtime/grid_util.h"

namespace physis {
namespace runtime {

Grid::~Grid() {
  delete data_buffer_[0];
  if (double_buffering_) delete data_buffer_[1];
  // below is not strictly necessary
  data_[0] = NULL;
  data_[1] = NULL;
}


void Grid::InitBuffer() {
  LOG_DEBUG() << "Initializing grid buffer\n";
  data_buffer_[0] = new BufferHost(num_dims_, num_elms_);
  data_buffer_[0]->Allocate(size_);
  if (double_buffering_) {
    data_buffer_[1] = new BufferHost(num_dims_, num_elms_);
    data_buffer_[1]->Allocate(size_);
  } else {
    data_buffer_[1] = data_buffer_[0];
  }
  data_[0] = (char*)data_buffer_[0]->Get();
  data_[1] = (char*)data_buffer_[1]->Get();
}

void Grid::Swap() {
  std::swap(data_[0], data_[1]);
  std::swap(data_buffer_[0], data_buffer_[1]);
}

void *Grid::GetAddress(const IntArray &indices) {
#ifdef PS_DEBUG
  PSAssert(_data() == buffer()->Get());
#endif
  return (void*)(_data() +
                 GridCalcOffset3D(indices, size())
                 * elm_size());
}

void Grid::Copyin(void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
}

void Grid::Copyout(void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
}

void Grid::Set(const IntArray &indices, const void *buf) {
  Copyin(GetAddress(indices), buf, elm_size());
}

void Grid::Get(const IntArray &indices, void *buf) {
  Copyout(buf, GetAddress(indices), elm_size());
}

bool Grid::AttributeSet(enum PS_GRID_ATTRIBUTE a) {
  return ((attr_ & a) != 0);
}

std::ostream &Grid::Print(std::ostream &os) const {
  os << "Grid {"
     << "}";
  return os;
}

template <class T>
int ReduceGrid(Grid *g, PSReduceOp op, T *out) {
  if (g->num_elms() == 0) return 0;
  //LOG_DEBUG() << "Op: " << op << "\n";
  boost::function<T (T, T)> func = GetReducer<T>(op);
  T *d = (T *)g->_data();
  T v = d[0];
  for (size_t i = 1; i < g->num_elms(); ++i) {
    v = func(v, d[i]);
  }
  //LOG_DEBUG() << "Reduce grid: " << v << "\n";
  *out = v;
  return g->num_elms();
}

int Grid::Reduce(PSReduceOp op, void *out) {
  int rv = 0;
  switch (type_) {
    case PS_FLOAT:
      rv = ReduceGrid<float>(this, op, (float*)out);
      break;
    case PS_DOUBLE:
      rv = ReduceGrid<double>(this, op, (double*)out);
      break;
    default:
      PSAbort(1);
  }
  return rv;
}

bool GridSpace::RegisterGrid(Grid *g) {
  g->id() = grid_counter_.next();
  grids_.insert(std::make_pair(g->id(), g));
  return true;
}

bool GridSpace::DeregisterGrid(Grid *g) {
  assert(isContained(grids_, g->id()));
  grids_.erase(g->id());
  return true;
}

Grid *GridSpace::FindGrid(int id) const {
  //assert(id >= 0 && (unsigned)id < grids_.size());
  Grid *g =
      physis::find<int, Grid*>(grids_, id, NULL);
  assert(g);
  return g;
}

void GridSpace::DeleteGrid(Grid *g) {
  DeregisterGrid(g);
  delete g;
}  

void GridSpace::DeleteGrid(int id) {
  Grid *g = FindGrid(id);
  assert(g);
  DeleteGrid(g);
}  


} // runtime
} // physis

