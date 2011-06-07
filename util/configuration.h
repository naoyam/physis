// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_UTIL_CONFIGURATION_H_
#define PHYSIS_UTIL_CONFIGURATION_H_

#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

#include "physis/physis_common.h"
#include "util/lua_loader.h"

namespace physis {
namespace util {

class Configuration {
 public:
  Configuration();
  virtual ~Configuration() {}
  virtual int LoadFile(const std::string &path);
  virtual std::ostream &print(std::ostream &os) const;
  
 protected:
  typedef std::string KeyDesc;
  std::map<int, KeyDesc> key_desc_map_;
  void AddKey(int key, const std::string &name) {
    key_desc_map_.insert(std::make_pair(key, name));
  }
  LuaTable tbl_;
  const LuaValue *LookupInternal(int key_id) const {
    const std::string &name = key_desc_map_.find(key_id)->second;
    if (!tbl_.HasKey(name)) return NULL;
    return tbl_.Find(name)->second;
  }
};

} // namespace util
} // namespace physis


inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::Configuration &conf) {
  return conf.print(os);
}


#endif /* PHYSIS_UTIL_CONFIGURATION_H_ */
