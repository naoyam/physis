// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/runtime.h"

#include <string>

using std::string;

namespace physis {
namespace runtime {

bool ParseOption(int *argc, char ***argv, const string &opt_name,
                 int num_additional_args, vector<string> &opts) {
  string opt_str[2] = {"-" + opt_name, "--" + opt_name};
  std::list<char *> argv_list;
  for (int i = 0; i < *argc; ++i) {
    argv_list.push_back((*argv)[i]);
  }
  FOREACH (it, argv_list.begin(), argv_list.end()) {
    string arg(*it);
    if (!(arg == opt_str[0] || arg == opt_str[1])) continue;
    for (int i = 0; i < num_additional_args+1; ++i) {
      opts.push_back(string(*it));
      it = argv_list.erase(it);
    }
    *argc = argv_list.size();
    ENUMERATE (idx, j, argv_list.begin(), argv_list.end()) {
      (*argv)[idx] = *j;
    }
    return true;
  }
  return false;
}

static int ParseProcDim(const string &s, IntArray &psize) {
  size_t pos = 0;
  int i = 0;
  while (true) {
    if (i == PS_MAX_DIM) {
      LOG_ERROR() << "Process dimension exceeds maximum allowed dimensionality"
                  << " (" << PS_MAX_DIM << ")\n";
      exit(1);
    }
    size_t next = s.find_first_of("x", pos);
    if (next == string::npos) {
      next = s.size();
    }
    string d = s.substr(pos, next-pos);
    int x = physis::toInteger(d);
    psize[i++] = x;
    if (next == s.size()) {
      break;
    }
    pos = next + 1;
  }
  return i;
}

int GetProcessDim(int *argc, char ***argv, IntArray &proc_size) {
  for (int i = 0; i < *argc; i++) {
    if (strcmp((*argv)[i], "-physis-proc") == 0 ||
        strcmp((*argv)[i], "--physis-proc") == 0) {
      char *pp = (*argv)[i+1];
      return ParseProcDim(string(pp), proc_size);
    }
  }
  return -1;
}


Runtime::Runtime(): gs_(NULL) {
}

Runtime::~Runtime() {
}

void Runtime::Init(int *argc, char ***argv, int grid_num_dims,
                   va_list vl) {
  // Set __ps_trace if physis-trace option is given
  __ps_trace = NULL;
  string opt_name = "physis-trace";
  vector<string> opts;
  if (ParseOption(argc, argv, opt_name, 0, opts)) {
      __ps_trace = stderr;
      LOG_INFO() << "Tracing enabled\n";
  }
}

} // namespace runtime
} // namespace physis
