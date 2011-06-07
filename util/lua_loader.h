// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_UTIL_LUA_LOADER_H_
#define PHYSIS_UTIL_LUA_LOADER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

#include "physis/physis_util.h"

struct lua_State;

namespace physis {
namespace util {

enum LuaType {LUA_BOOLEAN, LUA_NUMBER, LUA_STRING, LUA_TABLE};

class LuaValue;
class LuaString;
class LuaNumber;
class LuaBoolean;
class LuaTable;

class LuaValue {
  LuaType type_;
 public:
  LuaValue(LuaType type): type_(type) {}
  virtual ~LuaValue() {}
  LuaType type() const { return type_; }
  virtual std::ostream &print(std::ostream &os) const = 0;
  virtual LuaValue *clone() const = 0;
  virtual bool get(bool &v) const = 0;
  virtual bool get(std::string &v) const = 0;
  virtual bool get(double &v) const = 0;
  template <class T>
  LuaTable *getAsLuaTable() {
    if (type_ != LUA_TABLE) return NULL;
    return static_cast<LuaTable*>(this);
  }
  const LuaTable *getAsLuaTable() const;
};

class LuaBoolean: public LuaValue {
  bool v_;
 public:
  LuaBoolean(bool v): LuaValue(LUA_BOOLEAN), v_(v) {}
  virtual std::ostream &print(std::ostream &os) const;
  bool operator()() const { return v_; }
  virtual LuaValue *clone() const {
    return new LuaBoolean(v_);
  }
  virtual  bool get(bool &v) const {
    v = v_;
    return true;
  }
  virtual bool get(std::string &v) const { return false; }
  virtual bool get(double &v) const { return false; }
};

class LuaNumber: public LuaValue {
  double v_;
 public:
  LuaNumber(double v): LuaValue(LUA_NUMBER), v_(v) {}
  virtual std::ostream &print(std::ostream &os) const;
  virtual LuaValue *clone() const {
    return new LuaNumber(v_);
  }
  virtual  bool get(bool &v) const { return false; }
  virtual bool get(std::string &v) const { return false; }
  virtual bool get(double &v) const {
    v = v_;
    return true;
  }    
};

class LuaString: public LuaValue {
  std::string v_;
 public:
  LuaString(const std::string &v): LuaValue(LUA_STRING), v_(v) {}
  virtual std::ostream &print(std::ostream &os) const;
  virtual LuaValue *clone() const;
  virtual  bool get(bool &v) const { return false; }
  virtual bool get(std::string &v) const {
    v = v_;
    return true;
  }
  virtual bool get(double &v) const { return false; }  
};

class LuaTable: public LuaValue {
  typedef std::map<int, LuaValue*> IndexMapType;  
  typedef std::map<std::string, LuaValue*> KeyMapType;
  IndexMapType lst_;  
  KeyMapType tbl_;
 public:
  LuaTable(): LuaValue(LUA_TABLE) {}
  ~LuaTable() {
    FOREACH (it, lst_.begin(), lst_.end()) {
      delete it->second;
    }
    FOREACH (it, tbl_.begin(), tbl_.end()) {
      delete it->second;
    }
  }
  IndexMapType &lst() { return lst_; }
  const IndexMapType &lst() const { return lst_; }  
  KeyMapType &tbl() { return tbl_; }
  const KeyMapType &tbl() const { return tbl_; }  
  virtual std::ostream &print(std::ostream &os) const;
  KeyMapType::const_iterator Find(const std::string &key) const {
    return tbl_.find(key);
  }
  bool HasKey(const std::string &key) const {
    return tbl_.find(key) != tbl_.end();
  }
  bool HasKey(int key) const {
    return lst_.find(key) != lst_.end();
  }
  void Merge(const LuaTable &tbl);
  virtual LuaValue *clone() const;
  virtual  bool get(bool &v) const { return false; }
  virtual bool get(std::string &v) const { return false; }
  virtual bool get(double &v) const { return false; }
  virtual bool get(std::vector<double> &v) const;
};


class LuaLoader {
 public:
  LuaLoader() {}
  ~LuaLoader() {}
  LuaTable *LoadFile(const std::string &lua_file_path) const;
 private:
  LuaTable *LoadTable(lua_State *L, bool root) const;
  LuaString *LoadString(lua_State *L) const;
  LuaBoolean *LoadBoolean(lua_State *L) const;
  LuaNumber *LoadNumber(lua_State *L) const;
};


} // namespace util
} // namespace physis

inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::LuaBoolean &v) {
  return v.print(os);
}

inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::LuaString &v) {
  return v.print(os);
}

inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::LuaNumber &v) {
  return v.print(os);
}

inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::LuaTable &v) {
  return v.print(os);
}


#endif /* PHYSIS_UTIL_LUA_LOADER_H_ */
