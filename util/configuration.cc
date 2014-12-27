// Licensed under the BSD license. See LICENSE.txt for more details.

#include "util/configuration.h"

#include <iostream>

#include "util/lua_loader.h"
#include "physis/physis_util.h"

using std::string;

namespace physis {
namespace util {


Configuration::Configuration() {}

int Configuration::LoadFile(const std::string &path) {
  LuaLoader ll;
  LuaTable *tbl = ll.LoadFile(path);
  tbl_.Merge(*tbl);
  LOG_DEBUG() << "Current config: " << *this << "\n";
  return 0;
}

std::ostream &Configuration::print(std::ostream &os) const {
  StringJoin sj;
  FOREACH (it, key_desc_map_.begin(), key_desc_map_.end()) {
    const KeyDesc &key = it->second;
    if (tbl_.HasKey(key)) {
      tbl_.Find(key)->second->print(sj << key << ": ");
    }
  }
  return os << "{" << sj.str() << "}";
}

} // namespace util
} // namespace physis
