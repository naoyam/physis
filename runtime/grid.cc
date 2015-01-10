// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/grid.h"

#include "physis/physis_util.h"
#include "runtime/grid_util.h"

#include <iostream>
#include <fstream>

namespace physis {
namespace runtime {

Grid* Grid::Create(const __PSGridTypeInfo *type_info,
                   int num_dims, const IndexArray &size, int attr) {
  Grid *g = new Grid(type_info, num_dims, size, attr);
  g->InitBuffer();
  return g;
}

Grid* Grid::Create(PSType type, int elm_size, int num_dims,
                   const IndexArray &size, int attr) {
  __PSGridTypeInfo info = {type, elm_size, 0, NULL};
  return Create(&info, num_dims, size, attr);
}

Grid::Grid(const __PSGridTypeInfo *type_info,
           int num_dims, const IndexArray &size, int attr):
    type_(type_info->type), num_dims_(num_dims),
    num_elms_(size.accumulate(num_dims)),
    size_(size), data_buffer_(NULL), attr_(attr) {
  CopyTypeInfo(type_info_, *type_info);
}

Grid::~Grid() {
  DeleteBuffers();
}

void Grid::InitBuffer() {
  LOG_DEBUG() << "Initializing grid buffer\n";
  data_buffer_ = new BufferHost();
}

void Grid::DeleteBuffers() {
  if (data_buffer_) {
    delete data_buffer_;
    data_buffer_ = NULL;
  }
}

void *Grid::GetAddress(const IndexArray &indices) {
  return (void*)(idata() +
                 GridCalcOffset(indices, size(), num_dims())
                 * elm_size());
}
#if 0
void Grid::Copyin(void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
}

void Grid::Copyout(void *dst, const void *src, size_t size) {
  memcpy(dst, src, size);
}
#endif
void Grid::Set(const IndexArray &indices, const void *buf) {
  memcpy(GetAddress(indices), buf, elm_size());
}

void Grid::Get(const IndexArray &indices, void *buf) {
  memcpy(buf, GetAddress(indices), elm_size());
}

bool Grid::AttributeSet(enum PS_GRID_ATTRIBUTE a) {
  return ((attr_ & a) != 0);
}

std::ostream &Grid::Print(std::ostream &os) const {
  os << "Grid {"
     << "\ttype: " << ToString(type_)
     << "}";
  return os;
}

template <class T>
int ReduceGrid(Grid *g, PSReduceOp op, T *out) {
  if (g->num_elms() == 0) return 0;
  //LOG_DEBUG() << "Op: " << op << "\n";
  boost::function<T (T, T)> func = GetReducer<T>(op);
  T *d = (T *)g->data();
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
    case PS_INT:
      rv = ReduceGrid<int>(this, op, (int*)out);
      break;
    case PS_LONG:
      rv = ReduceGrid<long>(this, op, (long*)out);
      break;
    default:
      PSAssert(type_ == PS_USER);
      LOG_ERROR()  << "Reduction of user type not supported\n";
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

std::ostream &GridSpace::Print(std::ostream &os) const {
  os << "GridSpace {"
     << "}";
  return os;
}


#ifdef CHECKPOINTING_ENABLED
static string GetPath(size_t id) {
  char *ckpt_dir = getenv("PHYSIS_CHECKPOINT_DIR");
  return (ckpt_dir? string(ckpt_dir) : ".") + "/" + toString(id);
}

void Grid::SaveToFile(const void *buf, size_t len) const {
  string path = GetPath((size_t)this);
  LOG_DEBUG() << "Saving to file: "
              << path << "\n";
  std::ofstream ckpt_file(path.c_str(),
                          std::ios_base::out | std::ios_base::binary);
  ckpt_file.write((const char*)buf, len);
  ckpt_file.close();
}

void Grid::RestoreFromFile(void *buf, size_t len) {
  string path = GetPath((size_t)this);
  LOG_DEBUG() << "Reading from file: " << path << "\n";
  std::ifstream ckpt_file(path.c_str(),
                          std::ios_base::in | std::ios_base::binary);
  ckpt_file.read((char*)buf, len);
  ckpt_file.close();
  LOG_DEBUG() << "Reading done\n";
}

void Grid::Save() {
  Buffer *buf = data_buffer_[0];
  SaveToFile(buf->Get(), buf->GetLinearSize());
}

void Grid::Restore() {
  Buffer *buf = data_buffer_[0];
  RestoreFromFile(buf->Get(),  buf->GetLinearSize());
}

void GridSpace::Save() {
  FOREACH (it, grids_.begin(), grids_.end()) {
    it->second->Save();
  }
}

void GridSpace::Restore() {
  FOREACH (it, grids_.begin(), grids_.end()) {
    it->second->Restore();
  }
}
#endif
} // runtime
} // physis

