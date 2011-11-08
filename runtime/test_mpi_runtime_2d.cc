#define PHYSIS_MPI
#include "physis/physis.h"
#include "physis/physis_util.h"
#include "runtime/mpi_runtime.h"
#include "runtime/grid_mpi_debug_util.h"

#define N (4)
#define NDIM (2)

using namespace std;
using namespace physis::runtime;
using namespace physis;

void test1() {
  LOG_DEBUG() << "Test 1: No halo grid";
  PSVectorInt halo = {0, 0};
  PSVectorInt global_offset = {0, 0};
  PSVectorInt grid_size = {N, N};
  GridMPI *g = (GridMPI*)__PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, grid_size, 0,
                                        global_offset, 0);
  int gid = g->id();
  std::cerr << *g << "\n";

  float *data = (float*)g->_data();
  
  for (int i = 0; i < g->local_size().accumulate(g->num_dims()); ++i) {
    data[i] = i;
  }

  print_grid<float>(g, gs->my_rank(), std::cerr);

  gs->ExchangeBoundaries(gid, IntArray(halo),
                         IntArray(halo), false);

  print_grid<float>(g, gs->my_rank(), std::cerr);

  cerr << "Before delete\n";
  PSPrintInternalInfo(stderr);
  PSGridFree(g);
  
  cerr << "After delete\n";
  PSPrintInternalInfo(stderr);
}


void test2() {
  LOG_DEBUG() << "Test 2\n";
  PSVectorInt halo = {1, 1};
  PSVectorInt global_offset = {0, 0};
  PSVectorInt grid_size = {N, N};
  GridMPI *g = (GridMPI *)__PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, grid_size, 0,
                                         global_offset, 0);
  int gid = g->id();
  std::cerr << *g << "\n";

  float *data = (float*)g->_data();
  
  for (int i = 0; i < g->local_size().accumulate(g->num_dims()); ++i) {
    data[i] = i;
  }

  print_grid<float>(g, gs->my_rank(), std::cerr);

  gs->ExchangeBoundaries(gid, IntArray(halo),
                         IntArray(halo), false);

  print_grid<float>(g, gs->my_rank(), std::cerr);

  PSPrintInternalInfo(stderr);
}

void test3() {
  LOG_DEBUG() << "Test 3\n";  
  PSVectorInt halo = {1, 1};
  PSVectorInt global_offset = {1, 0};
  PSVectorInt grid_size = {N-1, N};
  GridMPI *g = (GridMPI*)__PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, grid_size, 0,
                                        global_offset, 0);
  int gid = g->id();
  std::cerr << *g << "\n";

  float *data = (float*)g->_data();
  
  for (int i = 0; i < g->local_size().accumulate(g->num_dims()); ++i) {
    data[i] = i;
  }

  //print_grid<float>(g, gs->my_rank_, std::cerr);

  if (true) gs->ExchangeBoundaries(gid, IntArray(halo),
                                   IntArray(halo), false);

  print_grid<float>(g, gs->my_rank(), std::cerr);

  PSPrintInternalInfo(stderr);
}

void test4() {
  LOG_DEBUG() << "Test 4: halo with diagonal\n";
  PSVectorInt halo = {1, 1};
  PSVectorInt global_offset = {0, 0};
  PSVectorInt grid_size = {N, N};
  GridMPI *g = (GridMPI*)__PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, grid_size, 0,
                                        global_offset, 0);
  int gid = g->id();
  std::cerr << *g << "\n";

  float *data = (float*)g->_data();
  
  for (int i = 0; i < g->local_size().accumulate(g->num_dims()); ++i) {
    data[i] = i;
  }

  print_grid<float>(g, gs->my_rank(), std::cerr);

  gs->ExchangeBoundaries(gid, IntArray(halo), IntArray(halo), true);

  print_grid<float>(g, gs->my_rank(), std::cerr);

  PSPrintInternalInfo(stderr);
}

void test5() {
  LOG_DEBUG() << "Test 5: Copyin and copyout";
  PSVectorInt global_offset = {0, 0};
  PSVectorInt grid_size = {N, N};
  int num_elms = N*N;
  GridMPI *g = (GridMPI*)__PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, grid_size, 0,
                                        global_offset, 0);

  float *idata = new float[num_elms];
  float *odata = new float[num_elms];
  for (int i = 0; i < num_elms; ++i) {
    idata[i] = i;
  }
  PSGridCopyin(g, idata);
  print_grid<float>(g, gs->my_rank(), std::cerr);
  
  PSGridCopyout(g, odata);
  
  for (int i = 0; i < num_elms; ++i) {
    if (idata[i] != odata[i]) {
      cerr << "Copyin and copyout failed; "
           << "Input: " << idata[i]
           << ", Output: " << odata[i] << std::endl;
      exit(1);
    }
  }

  PSGridFree(g);
  PSPrintInternalInfo(stderr);
  delete[] idata;
  delete[] odata;
}


int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, NDIM, N, N);
  PSPrintInternalInfo(stderr);

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "test1") == 0) {
      test1();
    } else if (strcmp(argv[i], "test2") == 0) {
      test2();
    } else if (strcmp(argv[i], "test3") == 0) {
      test3();
    } else if (strcmp(argv[i], "test4") == 0) {
      test4();
    } else if (strcmp(argv[i], "test5") == 0) {
      test5();
    }
  }

  PSFinalize();
  LOG_DEBUG() << "Finished\n";
  return 0;
}

