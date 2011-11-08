#define PHYSIS_MPI_CUDA
#include "physis/physis.h"
#include "physis/physis_util.h"
#include "runtime/mpi_cuda_runtime.h"
#include "runtime/grid_mpi_cuda_debug_util.h"
#include "runtime/mpi_util.h"

#define N (4)
#define NDIM (3)

using namespace std;
using namespace physis::runtime;
using namespace physis;

#define INDEX(i, j, k, N) (i + (j * N) + (k * N * N))

int my_rank;
float *make_grid(int nx, int ny, int nz) {
  float *buf = (float*)malloc(sizeof(float) * nx * ny * nz);
  int t = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        buf[t] = t;
        ++t;
      }
    }
  }
  return buf;
}

void set_grid(float *g, int nx, int ny, int nz, float v) {
  int t = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        g[t] = v;
        ++t;
      }
    }
  }
  return;
}

static void init_grid(GridMPI *g) {
  int idx = 0;
  float v = N * N * N* my_rank;
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




void test1(int *argc, char ***argv) {
  LOG_DEBUG() << "Test 1: Grid space creation and deletion\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  PSInit(argc, argv, NDIM, N, N, N, 0);
  PSFinalize();
  LOG_DEBUG() << "Test 1: Finished\n";
}

void test2(int *argc, char ***argv) {
  LOG_DEBUG() << "Test 2:  Grid creation and deletion\n";
  PSVectorInt global_size = {N, N, N};
  PSInit(argc, argv, NDIM, N, N, N, 0);
  __PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, global_size, false, NULL, 0);
  PSFinalize();
  LOG_DEBUG() << "Test 1: Finished\n";
}

void test3(int *argc, char ***argv) {
  LOG_DEBUG() << "Test 3:  Grid copyin and copyout";
  PSVectorInt global_size = {N, N, N};
  PSInit(argc, argv, NDIM, N, N, N, 0);
  __PSGridMPI * g =
      __PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, global_size, false, NULL, 0);
  float *indata = make_grid(N, N, N);
  float *outdata = make_grid(N, N, N);
  set_grid(outdata, N, N, N, 0.0);
  PSGridCopyin(g, indata);
  PSGridCopyout(g, outdata);
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < N; ++j) {
      for (int i= 0; i < N; ++i) {
        if (indata[INDEX(i, j, k, N)] != outdata[INDEX(i, j, k, N)]) {
          fprintf(stderr, "mismatch at %d,%d,%d; %f vs %f\n",
                  i, j, k, indata[INDEX(i, j, k, N)],
                  outdata[INDEX(i, j, k, N)]);
          exit(1);
        }
      }
    }
  }
  PSFinalize();
  LOG_DEBUG() << "Test 1: Finished\n";
}

void test4(int *argc, char ***argv) {
  LOG_DEBUG() << "Test 4:  ExchangeBoundaries\n"; 
  PSVectorInt global_size = {N, N, N};
  PSInit(argc, argv, NDIM, N, N, N, 0);
  __PSGridMPI * g =
      __PSGridNewMPI(PS_FLOAT, sizeof(float), NDIM, global_size, false, NULL, 0);
  float *indata = make_grid(N, N, N);
  float *outdata = make_grid(N, N, N);
  set_grid(outdata, N, N, N, 0.0);
  PSGridCopyin(g, indata);
  PSVectorInt halo = {1,1,1};
  __PSLoadNeighbor(g, halo, halo,  0, 0, 0);
  PSGridCopyout(g, outdata);
  print_grid<float>(static_cast<GridMPICUDA3D*>(g), 0, std::cout);
  PSFinalize();
  LOG_DEBUG() << "Test 4: Finished\n";
}

#if 0
void test4() {
  LOG_DEBUG_MPI() << "[" << my_rank << "] Test 4: Load subgrid self\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size,
                              false, global_offset);
  init_grid(g);
  
  // This should not perform any copy.
  GridMPI *g2 = gs->LoadSubgrid(*g, g->local_offset(), g->local_size());
  PSAssert(g2 == NULL);
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG_MPI() << "Test 4: Finished\n";
}

void test5() {
  LOG_DEBUG_MPI() << "Test 5: Overlapping LoadSubgrid\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  init_grid(g);
  
  IntArray goffset = g->local_offset() - 1;
  goffset.SetNoLessThan(0L);
  IntArray gy = g->local_offset() + g->local_size() + 1;
  gy.SetNoMoreThan(g->size());
  GridMPI *g2 = gs->LoadSubgrid(*g, goffset, gy - goffset);
  PSAssert(g2);
  print_grid<float>(g2, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG_MPI() << "Test 5: Finished\n";
}

void test6() {
  LOG_DEBUG_MPI() << "Test 6: LoadNeighbor no diagonal points\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  init_grid(g);

  GridMPI *g2 = gs->LoadNeighbor(*g, IntArray(1, 1, 1), IntArray(1, 1, 1), false);
  if (g2) {
    LOG_ERROR_MPI() << "Neighbor exchange not used\n";
    PSAbort(1);
  }
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG_MPI() << "Finished\n";
}

void test7() {
  LOG_DEBUG_MPI() << "LoadNeighbor with diagonal points\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  init_grid(g);

  GridMPI *g2 = gs->LoadNeighbor(*g, IntArray(1, 1, 1), IntArray(1, 1, 1), true);
  if (g2) {
    LOG_ERROR_MPI() << "Neighbor exchange not used\n";
    PSAbort(1);
  }
  print_grid<float>(g, my_rank, cerr);
  
  delete g;
  delete gs;
  LOG_DEBUG_MPI() << "Finished\n";
}

void test8() {
  LOG_DEBUG_MPI() << "Fetch same subgrid\n";
  IntArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  GridSpaceMPI *gs = new GridSpaceMPI(NDIM, global_size, NDIM, proc_size, my_rank);
  IntArray global_offset;
  GridMPI *g = gs->CreateGrid(sizeof(float), NDIM, global_size, false, global_offset);
  init_grid(g);
  IntArray goffset;
  LOG_DEBUG_MPI() << "offset: " << goffset << "\n";
  IntArray gsize(2, 2, 2);
  GridMPI *g2 = gs->LoadSubgrid(*g, goffset, gsize);
  if (my_rank == 0) {
    PSAssert(g2 == NULL);
  } else {
    PSAssert(g2);
    print_grid<float>(g2, my_rank, cerr);
  }
  
  delete g;
  delete gs;
  LOG_DEBUG_MPI() << "Finished\n";
}
#endif
int main(int argc, char *argv[]) {
  //assert(num_procs == 8);
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "test1") == 0) {
      test1(&argc, &argv);
    } else if (strcmp(argv[i], "test2") == 0) {
      test2(&argc, &argv);
    } else if (strcmp(argv[i], "test3") == 0) {
      test3(&argc, &argv);
    } else if (strcmp(argv[i], "test4") == 0) {
      test4(&argc, &argv);
#if 0                  
    } else if (strcmp(argv[i], "test5") == 0) {
      test5();
    } else if (strcmp(argv[i], "test6") == 0) {
      test6();
    } else if (strcmp(argv[i], "test7") == 0) {
      test7();
    } else if (strcmp(argv[i], "test8") == 0) {
      test8();
#endif      
    }
  }
  LOG_DEBUG_MPI() << "Finished\n";  
  return 0;
}

