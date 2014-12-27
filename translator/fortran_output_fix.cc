// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/fortran_output_fix.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

using std::ios_base;
using std::ifstream;
using std::ofstream;
using std::ostringstream;

namespace physis {
namespace translator {

// BEFORE: TYPE (PSStencil_kernel)
// FIXED:  TYPE (PSStencil_kernel), extends(PSStencil)
static bool FixStencilMapDecl(string &line, const vector<string> &tokens) {
  if (tokens.size() != 3) return false;  
  if (tokens[0] != "TYPE" ||
      tokens[1] != "::") return false;
  
  if (!startswith(tokens[2], "PSStencil_")) return false;

  LOG_DEBUG() << "Before: " << line << "\n";  
  line = tokens[0] + ", extends(PSStencil) " + tokens[1] + tokens[2];
  LOG_DEBUG() << "Fixed: " << line << "\n";

  return true;
  //return false;
}

// BEFORE: end type PSStencil_kernel
// FIXED: contains procedure run => PSStencilRun_kernel end type PSStencil_kernel
static bool FixStencilMapDeclTypeBoundProc(string &line, const vector<string> &tokens) {
  if (tokens.size() != 3) return false;
  if (tokens[0] != "END" ||
      tokens[1] != "TYPE") return false;

  string run_func_name = tokens[2];
  PSAssert(startswith(run_func_name, "PSStencil_"));
  boost::algorithm::replace_first(run_func_name, "PSStencil_", "PSStencilRun_");
  string proc_decl = "contains\nprocedure :: run => " + run_func_name + "\n";
  LOG_DEBUG() << "Before: " << line << "\n";      
  line = proc_decl + line;
  LOG_DEBUG() << "Fixed: " << line << "\n";        
  return true;
}

static bool FixStencilMapVarDecl(string &line, const vector<string> &tokens) {
  if (tokens.size() != 8) return false;
  if (tokens[0] != "TYPE" ||
      tokens[1] != "(" ||
      tokens[3] != ")" ||
      tokens[4] != "," ||
      tokens[5] != "POINTER" ||
      tokens[6] != "::" ||
      tokens[7] != "ps_stencil_p") return false;
  
  StringJoin sj(" ");
  sj << "CLASS (PSStencil)";
  BOOST_FOREACH (const string &s, make_pair(tokens.begin()+4, tokens.end())) {
    sj << s;
  }
  LOG_DEBUG() << "Before: " << line << "\n";    
  line = sj.str();
  LOG_DEBUG() << "Fixed: " << line << "\n";  
  return true;
}

static bool FixBaseStenciVarDecl(string &line, const vector<string> &tokens) {
  if (tokens.size() < 6) return false;
  if (tokens[0] != "TYPE" ||
      tokens[1] != "(" ||
      tokens[2] != "PSStencil" ||
      tokens[3] != ")" ||
      tokens[4] != "," ||
      tokens[5] != "POINTER") return false;

  
  StringJoin sj(" ");
  sj << "CLASS";
  BOOST_FOREACH (const string &s, make_pair(tokens.begin()+1, tokens.end())) {
    sj << s;
  }
  LOG_DEBUG() << "Before: " << line << "\n";    
  line = sj.str();
  LOG_DEBUG() << "Fixed: " << line << "\n";  
  return true;
}

// Assumption: within PSStencilRUn_kernel function
// Before: TYPE (PSStencil_kernel) :: s
// Fixed: CLASS (PSStencil_kernel) :: s
static bool FixRunStencilParamDecl(string &line, const vector<string> &tokens) {
  if (tokens.size() != 6) return false;
  if (tokens[0] != "TYPE" ||
      tokens[1] != "(" ||
      tokens[3] != ")" ||
      tokens[4] != "::" ||
      tokens[5] != "s") return false;

  string tn = tokens[2];
  if (!startswith(tn, "PSStencil_")) return false;


  StringJoin sj(" ");
  sj << "CLASS";
  BOOST_FOREACH (const string &s, make_pair(tokens.begin()+1, tokens.end())) {
    sj << s;
  }
  LOG_DEBUG() << "Before: " << line << "\n";    
  line = sj.str();
  LOG_DEBUG() << "Fixed: " << line << "\n";
  return true;
}  

static bool RunFuncBegin(const string &line, const vector<string> &tokens) {
  return startswith(line, "SUBROUTINE PSStencilRun_");
}

static bool SubroutineEnd(const string &line, const vector<string> &tokens) {
  return startswith(line, "END SUBROUTINE");
}

void FixFortranOutput(const string &path) {
  ifstream in(path.c_str());
  ostringstream os;
  string tmp;
  bool within_map_type = false;
  bool within_run_func = false;  
  while (!in.eof()) {
    std::getline(in, tmp, '\n');
    //std::cout << tmp << "\n";
    vector<string> tokens;
    boost::algorithm::split(tokens, tmp, boost::is_any_of(" "),
                            boost::algorithm::token_compress_on);
    if (FixStencilMapDecl(tmp, tokens)) {
      within_map_type = true;
    } else if (within_map_type && FixStencilMapDeclTypeBoundProc(tmp, tokens)) {
      within_map_type = false;
    } else if (FixStencilMapVarDecl(tmp, tokens)) {
    } else if (FixBaseStenciVarDecl(tmp, tokens)) {
    } else if (RunFuncBegin(tmp, tokens)) {
      within_run_func = true;
    } else if (SubroutineEnd(tmp, tokens)) {
      within_run_func = false;
    } else if (within_run_func && FixRunStencilParamDecl(tmp, tokens)) {
    } else {
      //LOG_DEBUG() << "Unchanged\n";
    }
    
    os << tmp << "\n";
    tmp.clear();
  }
  in.close();
  const string &fixed = os.str();
  ofstream out(path.c_str());
  //ofstream out((path + ".fixd").c_str());
  out.write(fixed.c_str(), fixed.size());
  out.close();
}

} // namespace translator
} // namespace physis

