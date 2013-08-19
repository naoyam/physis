// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/configuration.h"

namespace physis {
namespace translator {

/** auto tuning parameters: code pattern */
const std::string Configuration::at_params_pattern[] = {
  "OPT_KERNEL_INLINING",
  "OPT_LOOP_OPT",
  "OPT_LOOP_PEELING",
  "OPT_OFFSET_COMP",
  "OPT_OFFSET_CSE",
  "OPT_OFFSET_SPATIAL_CSE",
  "OPT_REGISTER_BLOCKING",
  "OPT_UNCONDITIONAL_GET",
  ""
};

/** auto tuning parameters: dynamic argument */
const std::string Configuration::at_params_dynamic[] = {
  "CUDA_BLOCK_SIZE",
  ""
};

int Configuration::LoadFile(const std::string &path, bool at) {
  int r = pu::Configuration::LoadFile(path);
  auto_tuning_ = false;
  ndynamic_ = 1;
  npattern_ = 0;
  int n = SetPat(0);
  if (!at) return r;
  npattern_ = n;
  if (npattern_ > 1) auto_tuning_ = true;
  /* dynamic argument */
  FOREACH (it, tbl_.tbl().begin(), tbl_.tbl().end()) {
    const pu::LuaTable *t = it->second->getAsLuaTable();
    if (!t) continue;
    for (int i = 0; !at_params_dynamic[i].empty(); ++i) {
      if (it->first == at_params_dynamic[i]) {
        if (it->first == "CUDA_BLOCK_SIZE") {
          /* Check nesting, e.g., "{{32, 4, 1}, {64, 4 , 1}} */
          if (!t->lst().begin()->second->getAsLuaTable()) {
            break;
          }
        }
        if (t->lst().size() > 1) {
          auto_tuning_ = true;
          ndynamic_ *= t->lst().size();
        }
        break;
      }
    }
  }
  return r;
}

/** set pattern & get number of patterns
 * @param[in] pat ... index of pattern
 * @return    number of patterns
 */
int Configuration::SetPat(int pat) {
  if ((npattern_ && pat >= npattern_) || pat < 0) return 0;
  tmptbl_.tbl().clear();
  int n = 1;
  FOREACH (it, tbl_.tbl().begin(), tbl_.tbl().end()) {
    const pu::LuaTable *t = it->second->getAsLuaTable();
    int f = 0;
    if (t) {
      for (int i = 0; !at_params_pattern[i].empty(); ++i) {
        if (it->first == at_params_pattern[i]) {
          int size = t->lst().size();
          pu::LuaValue *v = t->lst().find(1 + (pat % size))->second;
          tmptbl_.Insert(it->first, v);
          pat /= size;
          n *= size;
          f = 1;
          //LOG_INFO() << "SetPat: " << it->first << " = " << LookupFlag(it->first) << "\n";
          break;
        }
      }
    }
    if (!f) {
      tmptbl_.Insert(it->first, it->second);  /* copy */
    }
  }
  PSAssert(npattern_ <= 0 || npattern_ == n);
  return n;
}

/** print configuration */
std::ostream &Configuration::print(std::ostream &os) const {
  pu::Configuration::print(os);
  StringJoin sj;
  FOREACH (it, key_desc_map_.begin(), key_desc_map_.end()) {
    const KeyDesc &key = it->second;
    if (tmptbl_.HasKey(key)) {
      tmptbl_.Find(key)->second->print(sj << key << ": ");
    }
  }
  return os << ", AT: {" << sj.str() << "}";
}

} // namespace translator
} // namespace physis

