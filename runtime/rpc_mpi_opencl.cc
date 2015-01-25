// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/rpc_mpi_opencl.h"
#include "runtime/mpi_wrapper.h"
#include "runtime/grid_util.h"


namespace physis {
namespace runtime {
MasterMPIOpenCL::MasterMPIOpenCL(
    const ProcInfo &pinfo,
    GridSpaceMPIOpenCL *gs, MPI_Comm comm,
    CLbaseinfo *cl_in
                                 ):
    Master(pinfo, gs, comm),
    cl_generic_(cl_in)
{
  pinned_buf_ = new BufferOpenCLHost(0, 1);
}

MasterMPIOpenCL::~MasterMPIOpenCL() {
  delete pinned_buf_;

}

ClientMPIOpenCL::ClientMPIOpenCL(
    const ProcInfo &pinfo,
    GridSpaceMPIOpenCL *gs, MPI_Comm comm,
    CLbaseinfo *cl_in                
                                 ):
    Client(pinfo, gs, comm),
    cl_generic_(cl_in)
{
}

ClientMPIOpenCL::~ClientMPIOpenCL() {}

// Finalize
void MasterMPIOpenCL::Finalize() {
  //OpenCL_SAFE_CALL(openclThreadExit());
  static_cast<GridSpaceMPIOpenCL*>(gs_)->PrintLoadNeighborProf(std::cerr);  
  Master::Finalize();
}

void ClientMPIOpenCL::Finalize() {
  //OpenCL_SAFE_CALL(openclThreadExit());
  static_cast<GridSpaceMPIOpenCL*>(gs_)->PrintLoadNeighborProf(std::cerr);   
  Client::Finalize();
}

void MasterMPIOpenCL::GridCopyinLocal(GridMPI *g, const void *buf) {
  // extract the subgrid from buf for this process to the pinned
  // buffer
  size_t size = g->local_size().accumulate(g->num_dims()) *
      g->elm_size();
  pinned_buf_->EnsureCapacity(size);
  PSAssert(pinned_buf_->Get());
  CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                 g->size(), pinned_buf_->Get(),
                 g->local_offset(), g->local_size());
  // send it out to device memory
  static_cast<BufferOpenCLDev*>(g->buffer())->Copyin(
      *pinned_buf_, IndexArray(), g->local_size());
}

void MasterMPIOpenCL::GridCopyoutLocal(GridMPI *g, void *buf) {
  BufferOpenCLDev *dev_buf = static_cast<BufferOpenCLDev*>(g->buffer());
  size_t size = g->local_size().accumulate(g->num_dims()) *
      g->elm_size();
  PSAssert(dev_buf->GetLinearSize() == size);
  LOG_DEBUG() << "local_size:" << g->local_size() << "\n";
  dev_buf->Copyout(*pinned_buf_, IndexArray(), g->local_size());
  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), pinned_buf_->Get(), g->local_offset(),
                g->local_size());
}

} // namespace runtime
} // namespace physis
