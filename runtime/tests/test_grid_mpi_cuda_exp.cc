// Licensed under the BSD license. See LICENSE.txt for more details.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "runtime/grid_mpi_cuda_exp.h"
#include "runtime/grid_space_mpi_cuda.h"
#include "runtime/ipc_mpi.h"
#include "runtime/grid_mpi_debug_util.h"

#include <iostream>
#include <algorithm>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>

#define N (8)

using namespace ::testing;
using namespace ::std;
using namespace ::physis::runtime;
using namespace ::physis;

typedef GridMPICUDAExp GridType;
typedef GridSpaceMPICUDA<GridType> GridSpaceMPIType;

IntArray proc_size;
InterProcCommMPI *ipc;


template <class T>
T GMEMRead(T *p) {
  T h;
  cudaMemcpy(&h, p, sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

template <class T>
void GMEMWrite(T *p, T h) {
  cudaMemcpy(p, &h, sizeof(T), cudaMemcpyHostToDevice);
}

template <class T>
static void InitGrid(GridType *g) {
  if (g->num_dims() == 3) {
    std::cerr << "local_offset: " << g->local_offset()
              << ", local_size: " << g->local_size() << "\n";
    for (int k = 0; k < g->local_size()[2]; ++k) {
      for (int j = 0; j < g->local_size()[1]; ++j) {
        for (int i = 0; i < g->local_size()[0]; ++i) {
          IndexArray ijk = IndexArray(i, j, k);
          IndexArray t = ijk + g->local_offset();
          T v = t[0] + t[1] * g->size()[0]
              + t[2] * g->size()[0]
              * g->size()[1];
          //std::cerr << "write: " << v << "\n";
          GMEMWrite((T*)g->GetAddress(t), v);
          assert(v == GMEMRead((T*)g->GetAddress(t)));
        }
      }
    }
  } else if  (g->num_dims() == 2) {
    for (int j = 0; j < g->local_size()[1]; ++j) {    
      for (int i = 0; i < g->local_size()[0]; ++i) {
        IndexArray ij = IndexArray(i, j);
        IndexArray t = ij + g->local_offset();
        T v = t[0] + t[1] * g->size()[0];
        GMEMWrite((T*)g->GetAddress(t), v);
      }
    }
  } else if  (g->num_dims() == 1) {
    for (int i = 0; i < g->local_size()[0]; ++i) {
      IndexArray t = IndexArray(i+g->local_offset()[0]);
      T v = i;
      GMEMWrite((T*)g->GetAddress(t), v);
    }
  } else {
    LOG_ERROR() << "Unsupported dimension\n";
    exit(1);
  }
}

template <class T>
class Grid3DFloatTestBase: public T {
 public:
  virtual void SetUp() {
    IndexArray global_size(N, N, N);
    for (int i = 0; i < 3; ++i) {
      assert (proc_size[i] <= global_size[i]);
    }
    gs_ = new GridSpaceMPIType(
        3, global_size, 3, proc_size, *ipc);
    stencil_min_ = std::tr1::get<0>(this->GetParam());
    stencil_max_ = std::tr1::get<1>(this->GetParam());
    width_.fw = stencil_max_;
    width_.bw = stencil_min_ * -1;
    g_ = gs_->CreateGrid(
        PS_FLOAT, sizeof(float), 3, global_size,
        IndexArray(0), stencil_min_, stencil_max_, 0);
    InitGrid<float>(g_);
  }
  
  virtual void TearDown() {
    delete g_;
    delete gs_;
  }

  float Get(const IndexArray &idx) const {
    return GMEMRead((float*)g_->GetAddress(idx));
  }
  
  GridSpaceMPIType *gs_;
  GridType *g_;
  IndexArray stencil_min_;
  IndexArray stencil_max_;
  Width2 width_;
};

class Grid3DFloatCopyOutHaloTest:
    public Grid3DFloatTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, int, bool, bool> > > {};

TEST_P(Grid3DFloatCopyOutHaloTest, CopyOutHalo) {
  int dim = std::tr1::get<2>(GetParam());
  bool diag = std::tr1::get<3>(GetParam());
  bool fw = std::tr1::get<4>(GetParam());
  for (int i = 2; i > dim; --i) {
    gs_->ExchangeBoundaries(g_, i, width_, false, false);    
  }
  g_->CopyoutHalo(dim, width_, fw, diag);
  LOG_DEBUG() << "Copyout done\n";
  if (g_->halo()(dim, fw) > 0 &&
      ((fw && gs_->my_idx()[dim] != 0) ||
       (!fw && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1))) {
    float *buf = (float*)g_->GetHaloSelfHost(dim, fw)->Get();
    IndexArray idx_begin(0);
    IndexArray idx_end = g_->local_real_size();
    if (fw) {
      idx_begin[dim] = g_->halo()(dim, false);
    } else {
      idx_begin[dim] = g_->local_real_size()[dim] -
          g_->halo()(dim, true) - g_->halo()(dim, false);
    }
    idx_end[dim] = idx_begin[dim] + g_->halo()(dim, fw);
    int buf_idx = 0;
    IndexArray halo_size = g_->local_real_size();
    halo_size[dim] = g_->halo()(dim, fw);
    for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
      if (dim < 2) {
        // handle edge processe, which do not have halo unless
        // periodic
        if (k < (int)g_->halo()(2, false)) {
          if (gs_->my_idx()[2] == 0) {
            buf_idx += halo_size[0] * halo_size[1];
            continue;
          }
        } else if (k >= (int)g_->halo()(2, false) + g_->local_size()[2]) {
          if (gs_->my_idx()[2] == gs_->proc_size()[2] - 1) {
            break;
          }
        }
      }
      for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
        if (dim > 1) {
          if (j < (int)g_->halo()(1, false) ||
              j >= (int)g_->halo()(1, false) + g_->local_size()[1]) {
            buf_idx += g_->local_real_size()[0];
            continue;
          }
        } else if (dim < 1) {
          if ((j < (int)g_->halo()(1, false) &&
              gs_->my_idx()[1] == 0) ||
              (j >= (int)g_->halo()(1, false) + g_->local_size()[1] &&
               gs_->my_idx()[1] == gs_->proc_size()[1] - 1)) {
            buf_idx += halo_size[0];
            continue;
          }
        }
        for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
          if (dim > 0) {
            if (i < (int)g_->halo()(0, false) ||
                i >= (int)g_->halo()(0, false) + g_->local_size()[0]) {
              buf_idx += 1;
              continue;
            }
          }
          
          IndexArray ijk = IndexArray(i, j, k) + g_->local_real_offset();
          float v = ijk[0] + ijk[1] * g_->size()[0]
              + ijk[2] * g_->size()[0] * g_->size()[1];
          ASSERT_EQ(buf[buf_idx], v)
              << "(i,j,k) = (" << i << "," << j << "," << k << "), "
              << "buf idx: " << buf_idx
              << ", ijk: " << ijk
              << ", loacl_real_offset: " << g_->local_real_offset()
              << ", loacl_real_size: " << g_->local_real_size()
              << ", fw: " << fw
              << ", dim: " << dim;
          ++buf_idx;
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    EachDirectionNoDiag7pt, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -1, -1)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Range(0, 3),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    EachDirectionNoDiag13pt, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -2)),
        ::testing::Values(IndexArray(2, 2, 2)),
        ::testing::Range(0, 3),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    EachDirectionNoDiagAsymmetry, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -1, -2)),
        ::testing::Values(IndexArray(1, 2, 1)),
        ::testing::Range(0, 3),
        ::testing::Values(false),
        ::testing::Bool()));

// Copyout with diag for dimension 0 is not yet implemented
INSTANTIATE_TEST_CASE_P(
    EachDirectionWithDiag7pt, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -1, -1)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Range(1, 3),
        ::testing::Values(true),
        ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    EachDirectionWithDiag13pt, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -2)),
        ::testing::Values(IndexArray(2, 2, 2)),
        ::testing::Range(1, 3),
        ::testing::Values(true),
        ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    EachDirectionWithDiagAsymmetry, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -2)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Range(1, 3),
        ::testing::Values(true),
        ::testing::Bool()));


class Grid3DFloatExchangeBoundariesDimTest:
    public Grid3DFloatTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, int> > > {};


TEST_P(Grid3DFloatExchangeBoundariesDimTest, ExchangeBoundaries) {
  //g_->Print(std::cout);
  //std::cout << "\n";
  int dim = std::tr1::get<2>(GetParam());
  for (int i = 2; i >= dim; --i) {
    gs_->ExchangeBoundaries(g_, i, width_, false, false);
  }
  IndexArray idx_begin = g_->local_offset();
  IndexArray idx_end = g_->local_offset() + g_->local_size();
  float diff = 1;
  for (int i = 0; i < dim; ++i) {
    diff *= N;
  }
  idx_begin[dim] = g_->local_real_offset()[dim];
  idx_end[dim] = g_->local_offset()[dim];
  if (gs_->my_idx()[dim] != 0) {  
    for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
      for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
        for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
          IndexArray ijk = IndexArray(i, j, k);
          IndexArray ijk_ref = ijk;
          ijk_ref[dim] += 1;
          ASSERT_EQ(Get(ijk), Get(ijk_ref)-diff);
        }
      }
    }
  }
  idx_begin[dim] = g_->local_offset()[dim]+g_->local_size()[dim];
  idx_end[dim] = g_->local_real_offset()[dim]+g_->local_real_size()[dim];
  if (gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
    for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
      for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
        for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
          IndexArray ijk = IndexArray(i, j, k);
          IndexArray ijk_ref = ijk;
          ijk_ref[dim] -= 1;
          ASSERT_EQ(Get(ijk), Get(ijk_ref)+diff);
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    EachDimension7pt, Grid3DFloatExchangeBoundariesDimTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -1, -1)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Range(1, 3)));

INSTANTIATE_TEST_CASE_P(
    EachDimension13pt, Grid3DFloatExchangeBoundariesDimTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -2)),
        ::testing::Values(IndexArray(2, 2, 2)),
        ::testing::Range(1, 3)));

INSTANTIATE_TEST_CASE_P(
    EachDimensionAsymmetric, Grid3DFloatExchangeBoundariesDimTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -2, -2)),
        ::testing::Values(IndexArray(2, 1, 1)),
        ::testing::Range(1, 3)));

class Grid3DFloatExchangeBoundariesTest:
    public Grid3DFloatTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, bool, bool> > > {};

TEST_P(Grid3DFloatExchangeBoundariesTest, ExchangeBoundaries) {
  bool diag = std::tr1::get<2>(GetParam());
  bool periodic = std::tr1::get<3>(GetParam());
  gs_->ExchangeBoundaries(g_, 2, width_, diag, periodic);
  gs_->ExchangeBoundaries(g_, 1, width_, diag, periodic);
  gs_->ExchangeBoundaries(g_, 0, width_, diag, periodic);
  int dim = 2;
  IndexArray idx_begin = g_->local_real_offset();
  IndexArray idx_end = g_->local_real_offset() + g_->local_real_size();
  if (!periodic) {
    for (int i = 0; i < 3; ++i) {
      if (gs_->my_idx()[i] == 0) {
        idx_begin[i] +=  g_->halo()(i, false);
      } else if (gs_->my_idx()[i] == gs_->proc_size()[i]-1) {
        idx_end[i] -= g_->halo()(i, true);
      }
    }
  }
  for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
    for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int k = g_->local_real_offset()[2]; k < g_->local_offset()[2]; ++k) {
          ASSERT_EQ(
              Get(IndexArray(i, j, k)),
              Get(IndexArray(i, j, k+1)) - N*N);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int k = g_->local_offset()[2]+g_->local_size()[2];
             k < g_->local_real_offset()[2]+g_->local_real_size()[2]; ++k) {
          ASSERT_EQ(
              Get(IndexArray(i, j, k)),
              Get(IndexArray(i, j, k-1)) + N*N);
        }
      }
    }
  }
  dim = 1;
  for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
    for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int j = g_->local_real_offset()[dim]; j < g_->local_offset()[dim]; ++j) {
          ASSERT_EQ(Get(IndexArray(i, j, k)),
                    Get(IndexArray(i, j+1, k)) - N);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int j = g_->local_offset()[dim]+g_->local_size()[dim];
             j < g_->local_real_offset()[dim]+g_->local_real_size()[dim]; ++j) {
          ASSERT_EQ(
              Get(IndexArray(i, j, k)),
              Get(IndexArray(i, j-1, k)) + N);
        }
      }
    }
  }
  dim = 0;
  for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
    for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int i = g_->local_real_offset()[dim]; i < g_->local_offset()[dim]; ++i) {
          ASSERT_EQ(Get(IndexArray(i, j, k)),
                    Get(IndexArray(i+1, j, k))-1);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int i = g_->local_offset()[dim]+g_->local_size()[dim];
             i < g_->local_real_offset()[dim]+g_->local_real_size()[dim]; ++i) {
          ASSERT_EQ(
              Get(IndexArray(i, j, k)),
              Get(IndexArray(i-1, j, k)) + 1);
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    DiagonalPeriodic7pt, Grid3DFloatExchangeBoundariesTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -1, -1)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Bool(), ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    DiagonalPeriodic13pt, Grid3DFloatExchangeBoundariesTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -2)),
        ::testing::Values(IndexArray(2, 2, 2)),
        ::testing::Bool(), ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    DiagonalPeriodicAsymmetric, Grid3DFloatExchangeBoundariesTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -2)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Bool(), ::testing::Bool()));


int main(int argc, char *argv[]) {
  ::testing::InitGoogleMock(&argc, argv);
  ipc = InterProcCommMPI::GetInstance();
  ipc->Init(&argc, &argv);
  int proc_num_dims = GetProcessDim(&argc, &argv, proc_size);
  if (proc_num_dims == -1) {
    // if no process size specified
    proc_size = IntArray(1, 1, 1);
  }
  
  int x = RUN_ALL_TESTS();
  MPI_Finalize();
  return x;
}
