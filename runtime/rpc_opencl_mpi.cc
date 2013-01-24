#include <runtime/rpc_opencl_mpi.h>

/* C or C++ standard headers */
#include <cstdio>
#include <string>

#define SOURCE_ARRAY_SIZE  10
#define BUF_SIZE 1024

namespace physis {
namespace runtime {

CLMPIbaseinfo::CLMPIbaseinfo () :
    CLbaseinfo(), 
    header_path_(""), dev_id_(0), save_context_p(0)
{
} // CLMPIinfo

CLMPIbaseinfo::CLMPIbaseinfo (
    unsigned int id_use, unsigned int create_queue_p, unsigned int block_events_p) :
    CLbaseinfo(id_use, create_queue_p, block_events_p),
    header_path_(""), dev_id_(id_use), save_context_p(0)
{
} // CLMPIinfo

CLMPIbaseinfo::CLMPIbaseinfo (CLMPIbaseinfo &master) :
    CLbaseinfo(master.dev_id_, 0, master.cl_block_events_p)
{
  cldevid = master.cldevid;
  clcontext = master.clcontext;
  clqueue = 0;
  clkernel = 0;
  clprog = master.clprog;
  cl_num_platforms = master.cl_num_platforms;
  cl_pl_ids = master.cl_pl_ids;
  cl_num_devices = master.cl_num_devices;
  cl_dev_ids = master.cl_dev_ids;
  cl_block_events_p = master.cl_block_events_p;

  header_path_ = master.header_path_;
  kernel_filen_ = master.kernel_filen_;
  dev_id_ = master.dev_id_;

  save_context_p = 1;


  // Create queue
  clqueue = create_queue_from_context(dev_id_, cl_dev_ids, clcontext);
}

CLMPIbaseinfo::~CLMPIbaseinfo () {
  if (! save_context_p ) return;

  // Set clprog, clcontext, cl_dev_ids, cl_pl_ids to 0
  // before actually destroying them (i.e.
  // before calling clReleaseContext and so on)

  clprog = 0;
  clcontext = 0;
  cl_pl_ids = 0;
  cl_dev_ids = 0;

}

std::string CLMPIbaseinfo::create_kernel_contents(std::string kernelfile) const {
  std::string ret_str = "";
  int num = 0;

  // Write all header files to be included
  // The last element must be NULL


  const char *header_lists[] = {
    "physis/physis_mpi_opencl_device.h",
    NULL
  };

  // Add needed definitions
  ret_str += "#define PHYSIS_MPI_OPENCL\n";
  ret_str += "#define PHYSIS_MPI_OPENCL_KERNEL_MODE\n";
  ret_str += "#undef PHYSIS_USER\n";

  // For double usage
  // Disabled for AMD for now
#if 0
  ret_str += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
#endif

  // read header files and source file (kernelfile),
  // add the whole contents ret_str
  do {
    for (num = 0; num < SOURCE_ARRAY_SIZE - 1; num++) {
      FILE *fin;
      size_t size = 0;
      const char *file_read = NULL;

      int exit_loop = 0;
      std::string header_path = physis_opencl_h_include_path();


      if (header_lists[num] && (!(header_path.empty()))) {
        std::string str = header_path;
        str += "/";
        str += header_lists[num];
        file_read = str.c_str();
      } else {
        file_read = kernelfile.c_str();
        exit_loop = 1;
      }

      fin = fopen(file_read, "r");
      if (!fin) {
        fprintf(stderr, "fopen()ing file %s failed\n", file_read);
        break;
      }

      while (1) {
        char cbuf[BUF_SIZE];
        size = fread(cbuf, 1, BUF_SIZE - 1, fin);
        cbuf[size] = 0;
        ret_str += cbuf;
        if (size < BUF_SIZE - 1) break;
      }
      fclose(fin);

      // Insert new line
      ret_str += "\n";

      // exit this loop
      if (!header_lists[num]) {
        num++;
        break;
      }
      if (exit_loop) break;

    }; // for(num = 0; num < SOURCE_ARRAY_SIZE - 1; num++)

  } while(0);

  return ret_str;
} // create_kernel_contents

} // namespace runtime
} // namespace physis
