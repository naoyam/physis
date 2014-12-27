// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/stencil_range.h"
#include "translator/translation_util.h"
#include "translator/physis_exception.h"
#include <limits.h>

namespace physis {
namespace translator {

const std::string StencilIndexAttribute::name = "StencilIndexList";

void StencilIndexListInitSelf(StencilIndexList &sil, unsigned num_dims) {
  for (unsigned i = 1; i <= num_dims; ++i) {
    sil.push_back(StencilIndex(i, 0));
  }
}

void StencilIndexListClearOffset(StencilIndexList &sil) {
  FOREACH (it, sil.begin(), sil.end()) {
    it->offset = 0;
  }
}

int StencilIndexListFindDim(const StencilIndexList *sil, int dim) {
  ENUMERATE (i, it, sil->begin(), sil->end()) {
    if (it->dim == dim) return i;
  }
  return -1;
}

bool StencilIndexSelf(const StencilIndexList &sil, unsigned num_dims) {
  if (sil.size() != num_dims) return false;
  ENUMERATE (i, it, sil.begin(), sil.end()) {
    const StencilIndex &si = *it;
    if (si.dim != i+1 || si.offset != 0) return false;
  }
  return true;
}

bool StencilIndexRegularOrder(const StencilIndexList &sil,
                              unsigned num_dims) {
  if (sil.size() != num_dims) {
    // LOG_INFO() << "different dimension: "<< sil.size()
    //            << ", " << num_dims << "\n";
    return false;
  }
#if 0  
  ENUMERATE (i, it, sil.begin(), sil.end()) {
    const StencilIndex &si = *it;
    if (si.dim != i+1) return false;
  }
#else
  int cur_idx = sil.begin()->dim;
  ENUMERATE (i, it, sil.begin()+1, sil.end()) {
    const StencilIndex &si = *it;
    if (si.dim <= cur_idx) return false;
    cur_idx = si.dim;
  }
#endif  
  return true;
}

bool StencilIndexRegularOrder(const StencilIndexList &sil) {
  return StencilIndexRegularOrder(sil, sil.size());
}


// NOTE: dim starts from 1 instead of 0
void StencilRange::insert(int dim, const StencilIndex &si) {
  // min check
  StencilIndexList &min_ss = min_indices_[dim-1];
  bool compared = false;
  FOREACH (it, min_ss.begin(), min_ss.end()) {
    StencilIndex &cur_min = *it;
    if (cur_min.dim == si.dim) {
      if (si < cur_min) {
        cur_min = si;
      }
      compared = true;
      break;
    }
  }
  if (!compared) {
    min_ss.push_back(si);
  }
  // max check
  StencilIndexList &max_ss = max_indices_[dim-1];
  compared = false;
  FOREACH (it, max_ss.begin(), max_ss.end()) {
    StencilIndex &cur_max = *it;
    if (cur_max.dim == si.dim) {
      if (si > cur_max) {
        cur_max = si;
      }
      compared = true;
      break;
    }
  }
  if (!compared) {
    max_ss.push_back(si);
  }
  return;
}

void StencilRange::insert(const StencilIndexList &si) {
  PSAssert((unsigned)num_dims_ == si.size());
  int non_zero = 0;
  ENUMERATE (i, it, si.begin(), si.end()) {
    int dim = i+1;
    const StencilIndex &si = *it;
    insert(dim, si);
    if (si.offset) ++non_zero;
  }
  diagonal_ |= (non_zero > 1);
  all_indices_.push_back(si);
  return;
}

bool StencilRange::IsNeighborAccess() const {
  // NOTE: it should be just fine to check only minimum index
  for (int i = 0; i < num_dims_; ++i) {
    const StencilIndexList &idxlist = min_indices_[i];
    if (idxlist.size() != 1) {
      return false;
    }
    if (idxlist.begin()->dim != i+1) return false;
  }
  return true;
}

static string ToString(const StencilIndexList &sil) {
  StringJoin sj;
  FOREACH (it, sil.begin(), sil.end()) {
    sj << it->ToString();
  }
  return sj.str();
}

std::ostream &StencilRange::print(std::ostream &os) const {
  ostringstream ss;
  StringJoin bottom;
  StringJoin top;
  for (int i = 0; i < num_dims_; i++) {
    bottom << ToString(min_indices_[i]);
    top << ToString(max_indices_[i]);
  }
  ss << "(" << bottom << ")->(" << top << ")";
  return os << ss.str();
}

void StencilRange::merge(const StencilRange &sr) {
  LOG_DEBUG() << "Merge: self->" << *this << ", merged->" << sr << "\n";
  PSAssert(num_dims_ == sr.num_dims_);
  for (int i = 0; i < num_dims_; ++i) {
    FOREACH (it, sr.min_indices_[i].begin(), sr.min_indices_[i].end()) {
      insert(i+1, *it);
    }
    FOREACH (it, sr.max_indices_[i].begin(), sr.max_indices_[i].end()) {
      insert(i+1, *it);
    }
  }
  diagonal_ |= sr.diagonal_;
}

bool StencilRange::GetNeighborAccess(IntVector &offset_min, IntVector &offset_max) const {
  if (!IsNeighborAccess()) return false;

  for (int i = 0; i < num_dims_; ++i) {
    offset_min.push_back(min_indices_[i].begin()->offset);
    offset_max.push_back(max_indices_[i].begin()->offset);    
  }
  return true;
}

bool StencilRange::IsNeighborAccessDiagonalAccessed() const {
  if (!IsNeighborAccess()) return false;
  return diagonal_;
}

bool StencilRange::IsZero() const {
  // check no access
  if (!min_indices_[0].size()) return true;
  
  if (!IsNeighborAccess()) return false;

  for (int i = 0; i < num_dims_; ++i) {
    if (min_indices_[i].begin()->offset != 0) return false;
    if (max_indices_[i].begin()->offset != 0) return false;
  }
  
  return true;
}

SgVariableDeclaration *StencilRange::BuildPSGridRange(
    std::string name, SgScopeStatement *block) {
  // Build min_offsets
  __PSGridRange gr;
  gr.num_dims = num_dims_;
  for (int i = 0; i < num_dims_; ++i) {
    __PSOffsets &min_os = gr.min_offsets[i];
    min_os.num = min_indices_[i].size();
    ENUMERATE (j, it, min_indices_[i].begin(), min_indices_[i].end()) {
      min_os.offsets[j*2] = it->dim;
      min_os.offsets[j*2+1] = it->offset;
    }
    __PSOffsets &max_os = gr.max_offsets[i];
    max_os.num = max_indices_[i].size();
    ENUMERATE (j, it, max_indices_[i].begin(), max_indices_[i].end()) {
      max_os.offsets[j*2] = it->dim;
      max_os.offsets[j*2+1] = it->offset;
    }
  }
  return translator::BuildPSGridRange(name, block, gr);
}

bool StencilRange::IsUniqueDim() const {
  for (int i = 0; i < num_dims_; ++i) {
    if (min_indices_[i].size() != 1 ||
        max_indices_[i].size() != 1) {
      return false;
    }
  }
  return true;
}

int StencilRange::GetMaxWidth() const {
  ssize_t wd = -1;
  for (int i = 0; i < num_dims_; ++i) {
    FOREACH (it, min_indices_[i].begin(), min_indices_[i].end()) {
      wd = std::max(wd, -it->offset);
    }
    FOREACH (it, max_indices_[i].begin(), max_indices_[i].end()) {
      wd = std::max(wd, it->offset);
    }
  }
  LOG_DEBUG() << "Stencil max width: " << wd << "\n";
  PSAssert(wd < INT_MAX);
  return (int)wd;
}

ssize_t StencilRegularIndexList::GetIndex(int dim) const {
  PSAssert(isContained(indices_, dim));  
  map_t::const_iterator it = indices_.find(dim);
  return it->second;
}

void StencilRegularIndexList::SetIndex(int dim, ssize_t idx) {
  indices_[dim] = idx;
}

const std::string StencilIndexVarAttribute::name = "StencilIndexVar";

} // namespace translator
} // namespace physis
