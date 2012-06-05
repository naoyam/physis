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

int my_rank;

void test1() {
  LOG_DEBUG() << "[" << my_rank << "] Test 1: Grid space creation and deletion\n";
  IndexArray global_size;
  global_size[0] = N;  global_size[1] = N;
  IntArray proc_size;
  proc_size[0] = 2;  proc_size[1] = 2;
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 1: Finished\n";
}

void test2() {
  LOG_DEBUG() << "[" << my_rank << "] Test 2: Grid creation and deletion\n";
  IndexArray global_size;
  global_size[0] = N;  global_size[1] = N;
  IntArray proc_size;
  proc_size[0] = 2;  proc_size[1] = 2;
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IndexArray global_offset;
  GridMPI *g = gs->CreateGrid(PS_FLOAT, sizeof(float), NDIM, global_size, false,
                              global_offset, 0);
  int idx = 0;
  float v = N * N * my_rank;
  LOG_DEBUG() << "[" << my_rank << "] local size: " << g->local_size() << "\n";
  for (int i = 0; i < g->local_size()[0]; ++i) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      LOG_DEBUG() << "[" << my_rank << "] v: " << v << "\n";
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  }
  print_grid<float>(g, my_rank, cerr);
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 2: Finished\n";
}

void test3() {
  LOG_DEBUG() << "[" << my_rank << "] Test 3: Grid creation and deletion\n";
  IndexArray global_size;
  global_size[0] = N;  global_size[1] = N;
  IntArray proc_size;
  proc_size[0] = 2;  proc_size[1] = 2;
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IndexArray global_offset;
  GridMPI *g = gs->CreateGrid(PS_FLOAT, sizeof(float), NDIM, global_size,
                              false, global_offset, 0);
  int idx = 0;
  float v = N * N * my_rank;
  LOG_DEBUG() << "[" << my_rank << "] local size: " << g->local_size() << "\n";
  for (int i = 0; i < g->local_size()[0]; ++i) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      //LOG_DEBUG() << "[" << my_rank << "] v: " << v << "\n";
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  }
  UnsignedArray halo(1, 1);
  gs->ExchangeBoundaries(g->id(), halo, halo, false, false);
  print_grid<float>(g, my_rank, cerr);
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 3: Finished\n";
}

void test4() {
  LOG_DEBUG() << "[" << my_rank << "] Test 4: Load subgrid\n";
  IndexArray global_size;
  global_size[0] = N;  global_size[1] = N;
  IntArray proc_size;
  proc_size[0] = 2;  proc_size[1] = 2;
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IndexArray global_offset;
  GridMPI *g = gs->CreateGrid(PS_FLOAT, sizeof(float), NDIM, global_size,
                              false, global_offset, 0);
  int idx = 0;
  float v = N * N * my_rank;
  LOG_DEBUG() << "[" << my_rank << "] local size: " << g->local_size() << "\n";
  for (int i = 0; i < g->local_size()[0]; ++i) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      //LOG_DEBUG() << "[" << my_rank << "] v: " << v << "\n";
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  }
  
  // This should not perform any copy.
  GridMPI *g2 = gs->LoadSubgrid(g, g->local_offset(), g->local_size());
  if (g2) {
    LOG_DEBUG() << "Fetch performed\n";
  } else {
    LOG_DEBUG() << "Exchange performed\n"; 
  }
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 4: Finished\n";
}

void test5() {
  LOG_DEBUG() << "[" << my_rank << "] Test 5: Load subgrid\n";
  IndexArray global_size(N, N);
  IntArray proc_size(2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IndexArray global_offset;
  GridMPI *g = gs->CreateGrid(PS_FLOAT, sizeof(float), NDIM, global_size,
                              false, global_offset, 0);
  int idx = 0;
  float v = N * N * my_rank;
  LOG_DEBUG() << "[" << my_rank << "] local size: " << g->local_size() << "\n";
  for (int i = 0; i < g->local_size()[0]; ++i) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  }
  
  IndexArray goffset = g->local_offset() - IndexArray(1, 1);
  IndexArray gsize = g->local_size() + IndexArray(2, 2);

  GridMPI *g2 = gs->LoadSubgrid(g, goffset, gsize);
  PSAssert(g2 == NULL);
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 5: Finished\n";
}

void test6() {
  LOG_DEBUG() << "[" << my_rank << "] Test 6: Fetch subgrid\n";
  IndexArray global_size(N, N);
  IntArray proc_size(2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IndexArray global_offset;
  GridMPI *g = gs->CreateGrid(PS_FLOAT, sizeof(float), NDIM, global_size,
                              false, global_offset, 0);
  int idx = 0;
  float v = N * N * my_rank;
  LOG_DEBUG() << "[" << my_rank << "] local size: " << g->local_size() << "\n";
  for (int i = 0; i < g->local_size()[0]; ++i) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  }

  IndexArray goffset = g->local_offset();
  for (int i = 0; i < NDIM; ++i) {
    goffset [i] = (goffset[i] + N/2) % N;
  }
  LOG_DEBUG() << "[" << my_rank << "] offset: " << goffset << "\n";
  IndexArray gsize = g->local_size();
  GridMPI *g2 = gs->LoadSubgrid(g, goffset, gsize);
  PSAssert(g2);
  LOG_DEBUG() << "Fetch performed\n";
  print_grid<float>(g2, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 6: Finished\n";
}

void test7() {
  LOG_DEBUG() << "[" << my_rank << "] Test 7: Fetch subgrid\n";
  IndexArray global_size(N, N);
  IntArray proc_size(2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IndexArray global_offset;
  GridMPI *g = gs->CreateGrid(PS_FLOAT, sizeof(float), NDIM, global_size,
                              false, global_offset, 0);
  int idx = 0;
  float v = N * N * my_rank;
  LOG_DEBUG() << "[" << my_rank << "] local size: " << g->local_size() << "\n";
  for (int i = 0; i < g->local_size()[0]; ++i) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  }

  IndexArray goffset = g->local_offset();
  for (int i = 0; i < NDIM; ++i) {
    goffset [i] = (goffset[i] + N/2) % N;
  }
  if (goffset[0] >= N/2) {
    goffset[0]= (goffset[0] - N/4);
  } else {
    goffset[0]= (goffset[0] + N/4);
  }
  LOG_DEBUG() << "[" << my_rank << "] offset: " << goffset << "\n";
  IndexArray gsize = g->local_size();
  GridMPI *g2 = gs->LoadSubgrid(g, goffset, gsize);
  PSAssert(g2);
  LOG_DEBUG() << "Fetch performed\n";
  print_grid<float>(g2, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 7: Finished\n";
}

int main(int argc, char *argv[]) {
  LOG_INFO() << "This test code is likely to need adapation like the 3d version.\n";
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
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
    } else if (strcmp(argv[i], "test6") == 0) {
      test6();
    } else if (strcmp(argv[i], "test7") == 0) {
      test7();
    }
  }
  MPI_Finalize();
  LOG_DEBUG() << "[" << my_rank << "] Finished\n";
  return 0;
}

