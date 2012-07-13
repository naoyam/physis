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
  template <class T>
  bool Lookup(const std::string &key_name, T &value) const {
    //LOG_DEBUG() << "Lookup Key: " << key_name << "\n";
    if (!tbl_.HasKey(key_name)) return false;
    //LOG_DEBUG() << "Key exits: " << key_name << "\n";
    const LuaValue *lv = tbl_.Find(key_name)->second;
    lv->get(value);
    return true;
  }
  bool LookupFlag(const std::string &key_name) const {
    bool f = false;
    if (!Lookup<bool>(key_name, f)) return false;
    return f;
  }
  void SetFlag(const std::string &key_name, bool f) {
    LuaBoolean b(f);
    tbl_.Insert(key_name, &b);
    return;
  }
  
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
