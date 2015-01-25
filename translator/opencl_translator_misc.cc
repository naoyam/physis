// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/opencl_translator.h"
#include "translator/grid.h"
#include <cstdio>
#define BUFSIZE 10

namespace physis {
namespace translator {

// name_var_dom:
// For example name_var_dom(1, 0) returns std::string "__PS_dom_ymin"
std::string OpenCLTranslator::name_var_dom(int dim, int maxflag) {

  std::string ret = "__PS_dom_";
  ret += gridIndexNames[dim];
  if (maxflag)
    ret += "max";
  else
    ret += "min";

  return ret;

} // name_var_dom

// name_var_grid:
// For example name_var_grid(2, "foo") returns std::string "__PS_g2_foo"
std::string OpenCLTranslator::name_var_grid(int offset, std::string suffix) {
  char tmpstr[BUFSIZE];
  std::string ret;

  snprintf(tmpstr, BUFSIZE, "%i", offset);
  ret = "__PS_g";
  ret += tmpstr;
  ret += "_";
  ret += suffix;
  return ret;

} // name_var_grid

// name_var_grid_dim:
// For example, name_var_grid_dim(2, 2) returns
// std::string "__PS_g2_dim_y"
std::string OpenCLTranslator::name_var_grid_dim(int offset, int dim) {
  std::string ret;
  std::string suffix = "dim_";
  suffix += gridIndexNames[dim];
  return name_var_grid(offset, suffix);
} // name_var_grid_dim

// name_var_grid_dim
// For example, name_var_grid_dim("ga", 2) returns
// std::string "__PS_ga_dim_y"
std::string OpenCLTranslator::name_var_grid_dim(std::string oldname, int dim) {
  std::string ret = "__PS_";
  ret += oldname;
  ret += "_dim_";
  ret += gridIndexNames[dim];
  return ret;
}

} // namespace translator
} // namespace physis
