#define PHYSIS_MPI

#include "physis/physis.h"
#include "physis/physis_util.h"
#include "runtime/mpi_runtime2.h"
#include "runtime/grid_mpi_debug_util.h"


#define N (16)
#define NDIM (3)

using namespace std;
using namespace physis::runtime;
using namespace physis;

int test() {
  IndexArray global_size(N, N, N);
  IntArray proc_size(2, 2, 2);
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  Width2 stencil_width;
  stencil_width.fw = UnsignedArray(1, 1, 1);
  stencil_width.bw = UnsignedArray(1, 1, 1);  
  GridSpaceMPI2 *gs = new GridSpaceMPI2(NDIM,
                                        global_size,
                                        NDIM,
                                        proc_size,
                                        my_rank);
  GridMPI2 *g = gs->CreateGrid(PS_FLOAT, sizeof(float),
                               NDIM, global_size,
                               IndexArray(0),
                               stencil_width, 0);
  PSAssert(g);
  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  assert(test() == 0);
  MPI_Finalize();
  LOG_DEBUG() << "Finished\n";
  return 0;
}

