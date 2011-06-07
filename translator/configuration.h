// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_CONFIGURATION_H_
#define PHYSIS_TRANSLATOR_CONFIGURATION_H_

#include "util/configuration.h"

namespace pu = physis::util;

namespace physis {
namespace translator {

class Configuration: public pu::Configuration {
 public:
  enum ConfigKey {
    CUDA_PRE_CALC_GRID_ADDRESS,
    CUDA_BLOCK_SIZE,
    MPI_OVERLAP,
    MULTISTREAM_BOUNDARY};
  Configuration() {
    AddKey(CUDA_PRE_CALC_GRID_ADDRESS,
           "CUDA_PRE_CALC_GRID_ADDRESS");
    AddKey(CUDA_BLOCK_SIZE, "CUDA_BLOCK_SIZE");
    AddKey(MPI_OVERLAP, "MPI_OVERLAP");
    AddKey(MULTISTREAM_BOUNDARY, "MULTISTREAM_BOUNDARY");    
  }
  virtual ~Configuration() {}
  const pu::LuaValue *Lookup(ConfigKey key) const {
    return LookupInternal((int)key);
  }
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_CONFIGURATION_H_ */
