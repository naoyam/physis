#ifndef PHYSIS_RUNTIME_RPC_OPENCL_MPI_H
#define PHYSIS_RUNTIME_RPC_OPENCL_MPI_H

#include "runtime/rpc_opencl_common.h"

namespace physis {
namespace runtime {

class CLMPIbaseinfo : public CLbaseinfo {

 protected:
  virtual std::string create_kernel_contents(std::string kernelfile) const;
  virtual std::string physis_opencl_h_include_path(void) const { return header_path_; }

  std::string header_path_;
  std::string kernel_filen_;
  int dev_id_;
  int save_context_p;

 public:
  CLMPIbaseinfo();
  CLMPIbaseinfo(
      unsigned int id_default, unsigned int create_queue_p,
      unsigned int block_events_p);
  CLMPIbaseinfo(CLMPIbaseinfo &master);
  virtual ~CLMPIbaseinfo();

  virtual cl_program get_prog(void) const { return clprog; }
  virtual void set_kernel_filen(std::string filen) { kernel_filen_ = filen; }
  virtual std::string get_kernel_filen(void) const { return kernel_filen_; }

  virtual void set_header_include_path(const char *path) { if (path) header_path_ = path; }
  virtual void mark_save_context() { save_context_p = 1; }

  virtual void sync_queue() { if (clqueue) clFinish(clqueue); } 


}; // class CLMPIbaseinfo
} // namespace runtime
} // namespace physis


#endif /* #define PHYSIS_RUNTIME_RPC_OPENCL_MPI_H */
