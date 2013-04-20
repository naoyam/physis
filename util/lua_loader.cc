// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "util/lua_loader.h"

#include "physis/physis_util.h"
#include "physis/physis_common.h"


//#define LOG_DEBUG_LUA LOG_DEBUG
#define LOG_DEBUG_LUA LOG_NULL

using std::string;

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

static int check_lua(lua_State *ls, int status) {
  if (status) {
    LOG_ERROR() << lua_tostring(ls, -1) << "\n";
    lua_pop(ls, 1);
  }
  return status;
}

namespace physis {
namespace util {

std::ostream &LuaBoolean::print(std::ostream &os) const {
  if (v_) return os << "true";
  else return os << "false";
}

std::ostream &LuaNumber::print(std::ostream &os) const {
  return os << v_;
}

LuaValue *LuaString::clone() const {
  return new LuaString(v_);
}

std::ostream &LuaString::print(std::ostream &os) const {
  os << v_;
  return os;
}


std::ostream &LuaTable::print(std::ostream &os) const {
  StringJoin sj;
  FOREACH (it, tbl_.begin(), tbl_.end()) {
    it->second->print(sj << it->first << ": ");
  }
  FOREACH (it, lst_.begin(), lst_.end()) {
    it->second->print(sj << it->first << ": ");
  }
  os << "{" << sj.str() << "}";
  return os;
}

LuaValue *LuaTable::clone() const {
  LuaTable *nv = new LuaTable();
  nv->Merge(*this);
  return nv;
}

LuaTable *LuaLoader::LoadFile(const string &lua_file_path) const {
  // See http://csl.sublevel3.org/lua/ and
  // http://www.lua.org/pil/25.html
  
  LOG_DEBUG_LUA() << "Starting LUA interpreter\n";
  lua_State *L = lua_open();
  luaL_openlibs(L);

  LOG_DEBUG_LUA() << "Loading LUA file: " << lua_file_path << "\n";
  if (check_lua(L, luaL_loadfile(L, lua_file_path.c_str()))) {
    lua_close(L);
    PSAbort(1);
  }

  if (check_lua(L, lua_pcall(L, 0, 0, 0))) {
    lua_close(L);
    PSAbort(1);
  }

  LuaTable *tbl = LoadTable(L, true);
  
  lua_close(L);
  return tbl;
}

LuaString *LuaLoader::LoadString(lua_State *L) const {
  PSAssert(lua_isstring(L, -1));
  string s(lua_tostring(L, -1));
  LOG_DEBUG_LUA() << "Lua String: " << s << "\n";
  return new LuaString(s);
}

LuaBoolean *LuaLoader::LoadBoolean(lua_State *L) const {
  PSAssert(lua_isboolean(L, -1));
  bool b = lua_toboolean(L, -1);
  LOG_DEBUG_LUA() << "Lua boolean: " << b << "\n";
  return new LuaBoolean(b);
}

LuaNumber *LuaLoader::LoadNumber(lua_State *L) const {
  PSAssert(lua_isnumber(L, -1));
  double b = lua_tonumber(L, -1);
  LOG_DEBUG_LUA() << "Lua number: " << b << "\n";
  return new LuaNumber(b);
}

// For debugging
static void print_key(lua_State *L, int index)  {
  // Numeric check needs to be done earlier than string
  // check. lua_isstring returns true for numeric values since they
  // can be automatically convetted to strings in Lua.
  if (lua_isnumber(L, index)) {  
    int i = lua_tointeger(L, index);
    LOG_DEBUG_LUA() << "key (index): " << i << "\n";
  } else if (lua_isstring(L, index)) {
    string key(lua_tostring(L, index));
    LOG_DEBUG_LUA() << "key (string): " << key << "\n";
  } else {
    LOG_ERROR() << "Unsupported key type\n";
  }
  return;
}

static bool skip_entry(lua_State *L) {
  int key_index = -2;
  int val_index = -1;
  if (lua_istable(L, val_index)) {
    if (lua_isstring(L, key_index) &&
        !lua_isnumber(L, key_index)) {
      string key(lua_tostring(L, key_index));
      if (key == "package") return true;
      //if (key == "io") return true;
      //if (key == "os") return true;
      //if (key == "table") return true;
      //if (key == "debug") return true;
      //if (key == "math") return true;
      //if (key == "string") return true;
      //if (key == "coroutine") return true;
      //if (key == "") return true;
      if (key == "_G") return true;      
      //if (key == "CUDA_TRANSLATION_PRE_CALC_GRID_ADDRESS") return false;
      //return true;
    }
  }
  if (lua_isfunction(L, val_index)) {
    return true;
  }
  return false;
}

LuaTable *LuaLoader::LoadTable(lua_State *L, bool root) const {
  LuaTable *lt = new LuaTable();
  lua_pushnil(L);
  int tbl_index = root ? LUA_GLOBALSINDEX : -2;
  while (lua_next(L, tbl_index) != 0) {
    //LOG_DEBUG_LUA() << "next\n";
    if (skip_entry(L)) {
      lua_pop(L, 1);
      continue;
    }
    LOG_DEBUG_LUA() << lua_typename(L, lua_type(L, -2))
                    << " - "
                    << lua_typename(L, lua_type(L, -1)) << "\n";
    LuaValue *cv = NULL;
    if (lua_istable(L, -1)) {
      LOG_DEBUG_LUA() << "Nested table\n";
      print_key(L, -2);
      cv = LoadTable(L, false);
    } else if (lua_isboolean(L, -1)) {
      cv = LoadBoolean(L);
    } else if (lua_isnumber(L, -1)) {
      cv = LoadNumber(L);
    } else if (lua_isstring(L, -1)) {
      cv = LoadString(L);
    } else {
      LOG_DEBUG_LUA() << "Ignoring Unsupported table value\n";
    }

    if (!cv) {
      LOG_DEBUG_LUA() << "Entry not loaded\n";
    } else {
      if (lua_isnumber(L, -2)) {
        int index = lua_tointeger(L, -2);
        LOG_DEBUG_LUA() << "key (index): " << index << "\n";
        lt->lst().insert(std::make_pair(index, cv));
      } else if (lua_isstring(L, -2)) {
        string key(lua_tostring(L, -2));
        LOG_DEBUG_LUA() << "key (string): " << key << "\n";
        lt->tbl().insert(std::make_pair(key, cv));
      } else {
        LOG_ERROR() << "Unsupported key type\n";
        PSAbort(1);
      }
    }

    LOG_DEBUG_LUA() << "Entry processed (tbl_index: " << tbl_index << ")\n";

    /* removes 'value'; keeps 'key' for next iteration */
    lua_pop(L, 1);
  }
  LOG_DEBUG_LUA() << "Table loaded\n";
  return lt;
}

void LuaTable::Merge(const LuaTable &tbl) {
  FOREACH (it, tbl.tbl().begin(), tbl.tbl().end()) {
    PSAssert(!HasKey(it->first));
    LuaValue *nv = it->second->clone();
    tbl_.insert(std::make_pair(it->first, nv));
  }
  FOREACH (it, tbl.lst().begin(), tbl.lst().end()) {
    PSAssert(!HasKey(it->first));
    LuaValue *nv = it->second->clone();
    lst_.insert(std::make_pair(it->first, nv));
  }
  return;
}

bool LuaTable::get(std::vector<double> &v) const {
  FOREACH (it, lst_.begin(), lst_.end()) {
    double x;
    if (it->second->get(x)) {
      v.push_back(x);
    } else {
      return false;
    }
  }
  return true;
}

} // namespace util
} // namespace physis
