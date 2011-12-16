/* physis-OpenCL specific */

#define BUFSIZE 1024

/* C or C++ standard headers */
#include <cstring>
#include <cstdlib>
#include <cstdio>

/* physis-OpenCL specific*/
#include "runtime/rpc_opencl.h"

namespace physis {
  namespace runtime {

    void CLbaseinfo::guess_kernelfile
      (const int *argc, char ***argv, std::string &filename, std::string &kernelname) const {

      char buf[BUFSIZE];
      // const char *tail;
      const char *argi;
      char *pos;
      char **argvv = *argv;
      int found = 0;
      int i = 0;
      int len = 0;

      for (i = 0; i < *argc; i++) {
        argi = argvv[i];
        len = strlen(argi);
        if (len >= BUFSIZE - 4) continue;
        strncpy(buf, argi, len);
        pos = buf + len;
#if 0
        sprintf(pos, "%s", ".cl");
#else
        sprintf(pos, "%s", ".c");
#endif
        found = 1;
        break;
      }
      if (found)
        filename = buf;
      else
        filename = "";

      // At first set the below kernel name, will be updated by __PSSetKernel
      kernelname = "__PSStencilRun_kernel";
    } // guess_kernelfile

    std::string CLbaseinfo::physis_opencl_h_include_path(void) const {
      std::string ret = "";

      // FIXME
      // Currently no header files are included in kernel code

#if 0
      char buf[BUFSIZE];
      char *pos = NULL;

      snprintf(buf, BUFSIZE, "%s", __FILE__);
      pos = strrchr(buf, '/');
      if (!pos) return ret;
      *pos = 0;
      pos = strrchr(buf, '/');
      if (!pos) return ret;
      *pos = 0;
      ret = buf;
      ret += "/include";
#endif

      return ret;

    } // physis_opencl_h_dir_path

  } // namespace physis
} // namespace runtime
