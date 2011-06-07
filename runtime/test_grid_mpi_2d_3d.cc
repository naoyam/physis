#define PHYSIS_MPI
#include "physis/physis.h"
#include "physis/physis_util.h"
#include "runtime/mpi_runtime.h"
#include "runtime/grid_mpi_debug_util.h"

#define N (4)

using namespace std;
using namespace physis::runtime;
using namespace physis;

// dummy
typedef void (*StencilRunClientFunction)();
StencilRunClientFunction stencil_clients[10];;


int my_rank;

static void init_grid(GridMPI *g) {
  int idx = 0;
  float v = N * N * my_rank;
  if (g->num_dims() == 3) {
    for (int i = 0; i < g->local_size()[0]; ++i) {
      for (int j = 0; j < g->local_size()[1]; ++j) {
        for (int k = 0; k < g->local_size()[2]; ++k) {
          ((float*)(g->_data()))[idx] = v;
          ++v;
          ++idx;
        }
      }
    }
  } else if  (g->num_dims() == 2) {
    for (int i = 0; i < g->local_size()[0]; ++i) {
      for (int j = 0; j < g->local_size()[1]; ++j) {
        ((float*)(g->_data()))[idx] = v;
        ++v;
        ++idx;
      }
    }
  } else if  (g->num_dims() == 1) {
    for (int i = 0; i < g->local_size()[0]; ++i) {
      ((float*)(g->_data()))[idx] = v;
      ++v;
      ++idx;
    }
  } else {
    LOG_ERROR() << "Unsupported dimension\n";
    exit(1);
  }
}  

void test1() {
  LOG_DEBUG() << "[" << my_rank << "] "
              << __FUNCTION__ << ": Grid space creation and deletion\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(3, global_size, 3, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), 2, global_size,
                              false, global_offset, 0);
  init_grid(g);
  print_grid<float>(g, my_rank, cerr);  
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] " << __FUNCTION__ << ": Finished\n";  
}

void test2() {
  LOG_DEBUG() << "[" << my_rank << "] "
              << __FUNCTION__ << ": Grid space creation and deletion\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(3, global_size, 3, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), 2, global_size,
                              false, global_offset, 0);
  init_grid(g);
  
  if (my_rank >= 4) {
    IntArray os = g->local_offset();
    os[2] -= 2;
    GridMPI *g2 = gs->LoadSubgrid(g, os, g->local_size());
    PSAssert(g2);
    print_grid<float>(g2, my_rank, cerr);  
  } else {
    gs->LoadSubgrid(g, IntArray(), IntArray()); 
  }
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] " << __FUNCTION__ << ": Finished\n";
}


#if 0
void test2() {
  LOG_DEBUG() << "[" << my_rank << "] Test 2: Grid creation and deletion\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 2: Finished\n";
}

void test3() {
  LOG_DEBUG() << "[" << my_rank << "] Test 3: Grid creation and deletion\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  IntArray halo(1, 1, 1);
  gs->exchangeBoundaries(g->id(), halo, halo, false);
  print_grid<float>(g, my_rank, cerr);
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 3: Finished\n";
}

void test4() {
  LOG_DEBUG() << "[" << my_rank << "] Test 4: Load subgrid\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  // This should not perform any copy.
  GridMPI *g2 = gs->LoadSubgrid(*g, g->local_offset(), g->local_size());
  PSAssert(g2 == NULL);
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 4: Finished\n";
}

void test5() {
  LOG_DEBUG() << "[" << my_rank << "] Test 5: Load subgrid\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);  
  IntArray goffset = g->local_offset() - IntArray(1, 1);
  IntArray gsize = g->local_size() + IntArray(2, 2);

  GridMPI *g2 = gs->LoadSubgrid(*g, goffset, gsize);
  PSAssert(g2 == NULL);
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 5: Finished\n";
}

void test6() {
  LOG_DEBUG() << "[" << my_rank << "] Test 6: Fetch subgrid\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  IntArray goffset = g->local_offset();
  for (int i = 0; i < NDIM; ++i) {
    goffset [i] = (goffset[i] + N/2) % N;
  }
  LOG_DEBUG() << "[" << my_rank << "] offset: " << goffset << "\n";
  IntArray gsize = g->local_size();
  GridMPI *g2 = gs->LoadSubgrid(*g, goffset, gsize);
  PSAssert(g2);
  LOG_DEBUG() << "Fetch performed\n";
  print_grid<float>(g2, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 6: Finished\n";
}

void test7() {
  LOG_DEBUG() << "[" << my_rank << "] Test 7: Fetch subgrid\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  IntArray goffset = g->local_offset();
  for (int i = 0; i < NDIM; ++i) {
    goffset [i] = (goffset[i] + N/2) % N;
  }
  if (goffset[0] >= N/2) {
    goffset[0]= (goffset[0] - N/4);
  } else {
    goffset[0]= (goffset[0] + N/4);
  }
  LOG_DEBUG() << "[" << my_rank << "] offset: " << goffset << "\n";
  IntArray gsize = g->local_size();
  GridMPI *g2 = gs->LoadSubgrid(*g, goffset, gsize);
  PSAssert(g2);
  LOG_DEBUG() << "Fetch performed\n";
  print_grid<float>(g2, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG() << "[" << my_rank << "] Test 7: Finished\n";
}

#endif

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  assert(num_procs == 8);
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "test1") == 0) {
      test1();
    } else if (strcmp(argv[i], "test2") == 0) {
      test2();
#if 0            
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
#endif
    }
  }
  MPI_Finalize();
  LOG_DEBUG() << "[" << my_rank << "] Finished\n";
  return 0;
}

