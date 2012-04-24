// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_PHYSIS_UTIL_H_
#define PHYSIS_PHYSIS_UTIL_H_

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <assert.h>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <string.h>

#include "physis/config.h"

#define __FUNC_ID__ __FUNCTION__

//#define LOGGING_FILE_BASENAME(name)   (name)
#define LOGGING_FILE_BASENAME(name)                                     \
  ((std::string(name).find_last_of('/') == std::string::npos) ?         \
   std::string(name) : std::string(name).substr(std::string(name).find_last_of('/')+1))

#if defined(PS_VERBOSE)
#define LOG_VERBOSE()                             \
  (std::cerr << "[VERBOSE:" << __FUNC_ID__        \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)      \
   << "#"  << __LINE__ << "] ")
#else
#define LOG_VERBOSE()  if (0) std::cerr 
#endif

#if defined(PS_DEBUG)
#define LOG_DEBUG()                             \
  (std::cerr << "[DEBUG:" << __FUNC_ID__        \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)    \
   << "#"  << __LINE__ << "] ")
#else
#define LOG_DEBUG()  if (0) std::cerr 
#endif

#if defined(PS_WARNING)
#define LOG_WARNING()                           \
  (std::cerr << "[WARNING:" << __FUNC_ID__      \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)    \
   << "#"  << __LINE__ << "] ")
#else
#define LOG_WARNING() if (0) std::cerr
#endif

#define LOG_ERROR()                             \
  (std::cerr << "[ERROR:" << __FUNC_ID__        \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)    \
   << "#"  << __LINE__ << "] ")
#define LOG_INFO()                              \
  (std::cerr << "[INFO:" << __FUNC_ID__         \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)    \
   << "#"  << __LINE__ << "] ")

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
#define LOG_NULL() (std::ofstream("/dev/nulL"))
#elif defined(_WIN32)
#define LOG_NULL() (std::ofstream("nul"))
#else
#error Unsupported environment
#endif

#define FOREACH(it, it_begin, it_end)           \
  for (typeof(it_begin) it = (it_begin),        \
           _for_each_end = (it_end);            \
       it != _for_each_end; ++it)

#define ENUMERATE(i, it, it_begin, it_end)      \
  for (int i = 0; i == 0; ++i)                  \
    for (typeof(it_begin) it = (it_begin),      \
             _for_each_end = (it_end);          \
         it != _for_each_end; ++it, ++i)

#define FREE(p) do {                            \
    free(p);                                    \
    p = NULL;                                   \
  } while (0)

namespace physis {
using std::string;
using std::istringstream;
using std::ostringstream;
using std::vector;
using std::list;
using std::map;
using std::find;
using std::set;
    
inline string toString(int x) 
{
  ostringstream ss;
  ss << x;
  return ss.str();
}
    
inline string toString(unsigned x) 
{
  ostringstream ss;
  ss << x;
  return ss.str();
}
    
inline string toString(unsigned long x) 
{
  ostringstream ss;
  ss << x;
  return ss.str();
}
    
inline string toString(long  x) 
{
  ostringstream ss;
  ss << x;
  return ss.str();
}

/*
  inline string toString(const llvm::Type *x)  
  {
  ostringstream ss;
  ss << *x;
  return ss.str();
  }
    
  inline string toString(const llvm::Value *x)  
  {
  ostringstream ss;
  ss << *x;
  return ss.str();
  }
*/

template <class T>
bool isContained(const vector<T> &v, const T &x) 
{
  return std::find(v.begin(), v.end(), x) != v.end();
}
template <class T>
bool isContained(const list<T> &v, const T &x) 
{
  return std::find(v.begin(), v.end(), x) != v.end();
}

template <class T, class S>
bool isContained(const map<T, S> &v, const T &x) 
{
  return v.find(x) != v.end();
}

template <class T>
bool isContained(const set<T> &v, const T &x) 
{
  return v.find(x) != v.end();
}

inline int toInteger(const string &s)
{
  istringstream is(s);
  is.exceptions(istringstream::failbit |
                istringstream::badbit);
  int i;
  is >> i;
  return i;
}
    
inline unsigned toUnsignedInteger(const string &s)
{
  istringstream is(s);
  is.exceptions(istringstream::failbit |
                istringstream::badbit);
  unsigned i;
  is >> i;
  return i;
}
    
template<class T1, class T2>
inline T2 find(const std::map<T1, T2> &m, const T1 &k, const T2 &v) 
{
  typename std::map<T1, T2>::const_iterator it = m.find(k);
  if (it == m.end()) {
    return v;
  } else {
    return it->second;
  }
}

inline string getGenericName(const string &keyedName)
{
  size_t t1 = keyedName.find("_");
  size_t t2 = keyedName.find("_", t1+1);
  string genericName = keyedName.substr(0, t1+1)
      + keyedName.substr(t2+1);
  return genericName;
}

inline
bool startswith(const string &base, const string &prefix) 
{
  size_t found = base.find(prefix);
  return found != string::npos && found == 0;
}
    
inline
bool endswith(const string &base, const string &suffix) 
{
  size_t found = base.rfind(suffix);
  return found != string::npos &&
      found == (base.length() - suffix.length());
}

inline
string strip(const string &s) 
{
  string r;
  // http://www.cplusplus.com/reference/string/string/find_last_not_of/
  string ws(" \t\f\v\n\r");
  size_t found;
  found = s.find_last_not_of(ws);
  if (found == string::npos) {
    return string("");
  }
  r = s.substr(0, found+1);
  found = r.find_first_not_of(ws);
  assert (found != string::npos);
  r = r.substr(found, r.length() - found);
  return r;
}

class StringJoin
{
  const string sep;
  bool first;
  ostringstream ss;
 public:
  StringJoin(const string &sep=", "): sep(sep), first(true) {}
  void append(const string &s) {
    if (!first) ss << sep;
    ss << s;
    first = false;
  }
        
  template <class T>
  ostringstream &operator<< (const T &s) {
    if (!first) ss << sep;
    ss << s;
    first = false;
    return ss;
  }
  string get() const {
    return ss.str();
  }
  string str() const {
    return get();
  }
};

template <class I>
inline string toString(I first, I end) 
{
  StringJoin sj(", ");
  for (; first != end; first++) {
    sj << *first;
  }
  return "{" + sj.get() + "}";
}

class Counter
{
  unsigned v;
 public:
  Counter(unsigned origin=0): v(origin) {}
  unsigned next() {
    return v++;
  }
  unsigned read() {
    return v;
  }
};
} // namespace physis

inline std::ostream& operator<<(std::ostream &os,
                                const physis::StringJoin &sj) 
{
  return os << sj.get();
}



#endif /* PHYSIS_PHYSIS_UTIL_H_ */
