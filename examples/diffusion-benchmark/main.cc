#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <string>
#include <vector>
#include <map>


using std::vector;
using std::string;
using std::map;
using std::make_pair;

#include "diffusion3d.h"
#include "baseline.h"

#if defined(OPENMP)
#include "diffusion3d_openmp.h"
#endif

#if defined(OPENMP_TEMPORAL_BLOCKING)
#include "diffusion3d_openmp_temporal_blocking.h"
#endif

#if defined(CUDA) || defined(CUDA_OPT1) || defined(CUDA_OPT2) \
  || defined(CUDA_SHARED) || defined(CUDA_XY)
#include "diffusion3d_cuda.h"
#endif

#if defined(CUDA_TEMPORAL_BLOCKING)
#include "diffusion3d_cuda_temporal_blocking.h"
#endif

#if defined(MIC)
#include "diffusion3d_mic.h"
#endif

#if defined(PHYSIS)
#include "diffusion3d_physis.h"
#endif

using namespace diffusion3d;

#ifndef NX
#define NX (256)
#endif

void Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}

void PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--count N   " << "Number of iterations\n"
     << "\t--size N    "  << "Size of each dimension\n"
     << "\t--dump N    "  << "Dump the final data to file\n"
     << "\t--help      "  << "Display this help message\n";
}


void ProcessProgramOptions(int argc, char *argv[],
                           int &count, int &size,
                           bool &dump) {
  int c;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"count", 1, 0, 0},
      {"size", 1, 0, 0},
      {"dump", 0, 0, 0},
      {"help", 0, 0, 0},      
      {0, 0, 0, 0}
    };

    c = getopt_long(argc, argv, "",
                    long_options, &option_index);
    if (c == -1) break;
    if (c != 0) {
      //std::cerr << "Invalid usage\n";
      //PrintUsage(std::cerr, argv[0]);
      //Die();
      continue;
    }

    switch(option_index) {
      case 0:
        count = atoi(optarg);
        break;
      case 1:
        size = atoi(optarg);
        break;
      case 2:
        dump = true;
        break;
      case 3:
        PrintUsage(std::cerr, argv[0]);
        exit(EXIT_SUCCESS);
        break;
      default:
        break;
    }
  }
}

int main(int argc, char *argv[]) {

  int nx = NX; // default size
  int  count = 1000; // default iteration count
  bool dump = false;
  
  ProcessProgramOptions(argc, argv, count, nx, dump);
  Diffusion3D *bmk = NULL;

#if defined(OPENMP)
  bmk = new Diffusion3DOpenMP(nx, nx, nx);
#elif defined(OPENMP_TEMPORAL_BLOCKING)
  bmk = new Diffusion3DOpenMPTemporalBlocking(nx, nx, nx);
#elif defined(CUDA)
  bmk = new Diffusion3DCUDA(nx, nx, nx);
#elif defined(CUDA_OPT1)
  bmk = new Diffusion3DCUDAOpt1(nx, nx, nx);
#elif defined(CUDA_OPT2)
  bmk = new Diffusion3DCUDAOpt2(nx, nx, nx);
#elif defined(CUDA_SHARED)
  bmk = new Diffusion3DCUDAShared(nx, nx, nx);
#elif defined(CUDA_XY)
  bmk = new Diffusion3DCUDAXY(nx, nx, nx);
#elif defined(CUDA_TEMPORAL_BLOCKING)
  bmk = new Diffusion3DCUDATemporalBlocking(nx, nx, nx);
#elif defined(MIC)
  bmk = new Diffusion3DMIC(nx, nx, nx);
#elif defined(PHYSIS)
  bmk = new Diffusion3DPhysis(nx, nx, nx, argc, argv);
#else
  bmk = new Baseline(nx, nx, nx);
#endif
  
  bmk->RunBenchmark(count, dump);

  return 0;
}
