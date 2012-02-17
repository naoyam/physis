/* C or C++ standard headers */
#include <cstdarg>
#include <cstdio>

#include <algorithm>
#include <string>

/* physis-OpenCL specific headers*/
#include "runtime/opencl_runtime.h"

namespace physis {
  namespace runtime {
    CLinfo *master;
  } // namespace runtime
} // namespace physis


extern "C" {

  void PSInit(int *argc, char ***argv, int grid_num_dims, ...) {

    std::string filen, kerneln;

    // common initialization
    physis::runtime::PSInitCommon(argc, argv);
    // Initiaize CLinfo
    physis::runtime::master = new physis::runtime::CLinfo();

    // TODO: for now try this
    physis::runtime::master->guess_kernelfile(argc, argv, filen, kerneln);
    physis::runtime::master->create_program(filen);
    
  } // void PSInit()

  void PSFinalize(){
    delete physis::runtime::master;
    // FIXME
    // Why need this??
    fflush(stdout);
  } // void PSFinalize()

  // TODO: This function must need checking later, however
  // like other languages just return 0 for now 
  // (runtime source has a comment that Id is not used on shared memory)
  //
  int __PSGridGetID(__PSGrid *g) {
    return 0;
  } // void PSFinalize()


  // PSGrid3DRealNew(NX, NX, NX); is translated into
  // int dims[3] = {((index_t )NX), ((index_t )NX), ((index_t )NX)};
  // g = __PSGridNew(sizeof(float), 3, dims, 0);
  __PSGrid* __PSGridNew(int elm_size, int num_dims, PSVectorInt dim,
                        int grid_attribute) {
    return physis::runtime::master->GridNew(elm_size, num_dims, dim, grid_attribute);
  } // _PSGrid* __PSGridNew()

  void PSGridFree(void *p) {
    __PSGrid *g = (__PSGrid *)p;
    physis::runtime::master->GridFree(g);
  } // void PSGridFree()

  void PSGridCopyin(void *p, const void *src_array) {
    __PSGrid *g = (__PSGrid *)p;
    physis::runtime::master->GridCopyin(g, src_array);
  } // void PSGridCopyin()

  void PSGridCopyout(void *p, void *dst_array) {
    __PSGrid *g = (__PSGrid *)p;
    physis::runtime::master->GridCopyout(g, dst_array);
  } // void PSGridCopyout()

  void __PSGridSwap(__PSGrid *g) {
    // Currently double buffering is not used, so do nothing.
  } // void __PSGridSwap()

  PSDomain1D PSDomain1DNew(index_t minx, index_t maxx) {
    PSDomain1D d = {{minx}, {maxx}, {minx}, {maxx}};
    return d;
  } // PSDomain1D PSDomain1DNew()

  PSDomain2D PSDomain2DNew(index_t minx, index_t maxx,
                           index_t miny, index_t maxy) {
    PSDomain2D d = {{minx, miny}, {maxx, maxy},
                    {minx, miny}, {maxx, maxy}};
    return d;
  } // PSDomain2D PSDomain2DNew()

  PSDomain3D PSDomain3DNew(index_t minx, index_t maxx,
                           index_t miny, index_t maxy,
                           index_t minz, index_t maxz) {
    PSDomain3D d = {{minx, miny, minz}, {maxx, maxy, maxz},
                    {minx, miny, minz}, {maxx, maxy, maxz}};
    return d;
  } // PSDomain3D PSDomain3DNew()

  void PSReduce(void *v, ...){
    fprintf(stderr, "PSReduce is not implemented in OpenCL yet\n");
  } // PSReduce

  // For example, { float val; PSGridSet(grid_mat, xx, yy, zz, val); } is translated
  // into { float val; __PSGridSet(grid_mat, &val, xx, yy, zz); }

  // The variable arguments express the coordinate where the new value will be
  // set.
  void __PSGridSet(__PSGrid *g, const void *ptr_val, ...) {
    va_list valst;
    va_start(valst, ptr_val);
    physis::runtime::master->GridSet(g, ptr_val, valst);
    va_end(valst);
  } // void __PSGridSet()

  void __PSSetKernel(const char *kernelname) {
    std::string string_kernelname = kernelname;
    physis::runtime::master->create_kernel(string_kernelname);
  } // __PSSetKernel

  void __PSSetKernelArg(unsigned int arg_index, size_t arg_size, const void *arg_val) {
    physis::runtime::master->SetKernelArg((cl_uint) arg_index, arg_size, arg_val);
  } // __PS_SetKernelArg

  void __PSSetKernelArgCLMem(unsigned int arg_index, const void *arg_val) {
    physis::runtime::master->SetKernelArg((cl_uint) arg_index, sizeof(cl_mem), arg_val);
  } // __PS_SetKernelArg

  void __PSRunKernel(size_t *globalsize, size_t *localsize) {
    physis::runtime::master->RunKernel(globalsize, localsize);
  } // __PSRunKernel

} // extern "C" {}

