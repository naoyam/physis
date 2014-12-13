// Licensed under the BSD license. See LICENSE.txt for more details.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "runtime/grid_space_mpi.h"
#include "runtime/ipc_mpi.h"
#include "runtime/grid_mpi_debug_util.h"

#include <iostream>
#include <algorithm>
#include <tuple>

#define N (8)

using namespace ::testing;
using namespace ::std;
using namespace ::physis::runtime;
using namespace ::physis;

typedef GridSpaceMPI<GridMPI> GridSpaceMPIType;

IntArray proc_size;
InterProcCommMPI *ipc;

template <class T>
static void InitGrid(GridMPI *g) {
  if (g->num_dims() == 3) {
    for (int k = 0; k < g->local_size()[2]; ++k) {
      for (int j = 0; j < g->local_size()[1]; ++j) {
        for (int i = 0; i < g->local_size()[0]; ++i) {
          IndexArray ijk = IndexArray(i, j, k);
          IndexArray t = ijk + g->local_offset();
          T v = t[0] + t[1] * g->size()[0]
              + t[2] * g->size()[0]
              * g->size()[1];
          *((T*)(g->GetAddress(t))) = v;
        }
      }
    }
  } else if  (g->num_dims() == 2) {
    for (int j = 0; j < g->local_size()[1]; ++j) {    
      for (int i = 0; i < g->local_size()[0]; ++i) {
        IndexArray ij = IndexArray(i, j);
        IndexArray t = ij + g->local_offset();
        T v = t[0] + t[1] * g->size()[0];
        *((T*)(g->GetAddress(t))) = v;
      }
    }
  } else if  (g->num_dims() == 1) {
    for (int i = 0; i < g->local_size()[0]; ++i) {
      *((T*)(g->GetAddress(IndexArray(i+g->local_offset()[0])))) = i;
    }
  } else {
    LOG_ERROR() << "Unsupported dimension\n";
    exit(1);
  }
}  

#if 0
// Multiple inheritance does not work
class Grid3DFloatTest: public ::testing::Test {
 public:
  virtual void SetUp() {
    IndexArray global_size(N, N, N);
    for (int i = 0; i < 3; ++i) {
      assert (proc_size[i] <= global_size[i]);
    }
    gs_ = new GridSpaceMPIType(
        3, global_size, 3, proc_size, *ipc);
    g_ = gs_->CreateGrid(
        PS_FLOAT, sizeof(float), 3, global_size,
        IndexArray(0), IndexArray(-1, -1, -1), IndexArray(1, 1, 1), 0);
    InitGrid<float>(g_);
  }
  
  virtual void TearDown() {
    delete g_;
    delete gs_;
  }
  
  GridSpaceMPIType *gs_;
  GridMPI *g_;
};
#endif

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
        IndexArray(0), stencil_min_,
        stencil_max_, 0);
    InitGrid<float>(g_);
  }
  
  virtual void TearDown() {
    delete g_;
    delete gs_;
  }

  float Get(const IndexArray &idx) const {
    return *(float*)(g_->GetAddress(idx));
  }

  float &Get(const IndexArray &idx) {
    return *(float*)(g_->GetAddress(idx));
  }
  
  GridSpaceMPIType *gs_;
  GridMPI *g_;
  IndexArray stencil_min_;
  IndexArray stencil_max_;
  Width2 width_;
};

class Grid3DFloatCopyOutHaloTest:
    public Grid3DFloatTestBase< ::testing::TestWithParam< tr1::tuple<IndexArray, IndexArray, int, bool> > > {};

TEST_P(Grid3DFloatCopyOutHaloTest, CopyOutHalo) {
  int dim = std::tr1::get<2>(GetParam());
  bool diag = true;
  bool fw = std::tr1::get<3>(GetParam());
  for (int i = 2; i > dim; --i) {
    g_->CopyoutHalo(i, width_, true, diag);
    g_->CopyoutHalo(i, width_, false, diag);
  }
  g_->CopyoutHalo(dim, width_, fw, diag);
  if (g_->halo()(dim, fw) > 0) {
    float *buf = (float*)g_->GetHaloSelf(dim, fw);
    IndexArray idx_begin(0);
    IndexArray idx_end = g_->local_real_size();
    if (fw) {
      idx_begin[dim] = g_->halo()(dim, false);
    } else {
      idx_begin[dim] = g_->local_real_size()[dim]-
          g_->halo()(dim, true) - g_->halo()(dim, false);
    }
    idx_end[dim] = idx_begin[dim] + g_->halo()(dim, fw);
    int buf_idx = 0;
    for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
      for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
        for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
          EXPECT_EQ(buf[buf_idx],
                    Get(IndexArray(i, j, k) + g_->local_real_offset()));
          ++buf_idx;
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    EachDirection7pt, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(::testing::Values(IndexArray(-1, -1, -1)),
                       ::testing::Values(IndexArray(1, 1, 1)),
                       ::testing::Range(0, 3), ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    EachDirection13pt, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(::testing::Values(IndexArray(-2, -2, -2)),
                       ::testing::Values(IndexArray(2, 2, 2)),
                       ::testing::Range(0, 3), ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    EachDirectionAsymmetry, Grid3DFloatCopyOutHaloTest,
    ::testing::Combine(::testing::Values(IndexArray(-1, -2, 0)),
                       ::testing::Values(IndexArray(2, 1, 1)),
                       ::testing::Range(0, 3), ::testing::Bool()));


class Grid3DFloatExchangeBoundariesDimTest:
    public Grid3DFloatTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, int> > >{};

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
          EXPECT_EQ(Get(ijk), Get(ijk_ref)-diff);
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
          EXPECT_EQ(Get(ijk), Get(ijk_ref)+diff);
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    EachDimension7pt, Grid3DFloatExchangeBoundariesDimTest,
    ::testing::Combine(::testing::Values(IndexArray(-1, -1, -1)),
                       ::testing::Values(IndexArray(1, 1, 1)),
                       ::testing::Range(0, 3)));

INSTANTIATE_TEST_CASE_P(
    EachDimension13pt, Grid3DFloatExchangeBoundariesDimTest,
    ::testing::Combine(::testing::Values(IndexArray(-2, -2, -2)),
                       ::testing::Values(IndexArray(2, 2, 2)),
                       ::testing::Range(0, 3)));

INSTANTIATE_TEST_CASE_P(
    EachDimensionAsymmetry, Grid3DFloatExchangeBoundariesDimTest,
    ::testing::Combine(::testing::Values(IndexArray(-2, -1, -2)),
                       ::testing::Values(IndexArray(1, 2, 1)),
                       ::testing::Range(0, 3)));


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
        idx_begin[i] += g_->halo()(i, false);
      } else if (gs_->my_idx()[i] == gs_->proc_size()[i]-1) {
        idx_end[i] -= g_->halo()(i, true);
      }
    }
  }
  for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
    for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int k = g_->local_real_offset()[2]; k < g_->local_offset()[2]; ++k) {
          EXPECT_EQ(
              Get(IndexArray(i, j, k)),
              Get(IndexArray(i, j, k+1)) - N*N);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int k = g_->local_offset()[2]+g_->local_size()[2];
             k < g_->local_real_offset()[2]+g_->local_real_size()[2]; ++k) {
          EXPECT_EQ(
              Get(IndexArray(i, j, k)), Get(IndexArray(i, j, k-1)) + N*N);
        }
      }
    }
  }
  dim = 1;
  for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
    for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int j = g_->local_real_offset()[dim]; j < g_->local_offset()[dim]; ++j) {
          EXPECT_EQ(Get(IndexArray(i, j, k)),
                    Get(IndexArray(i, j+1, k)) - N);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int j = g_->local_offset()[dim]+g_->local_size()[dim];
             j < g_->local_real_offset()[dim]+g_->local_real_size()[dim]; ++j) {
          EXPECT_EQ(
              Get(IndexArray(i, j, k)), Get(IndexArray(i, j-1, k)) + N);
        }
      }
    }
  }
  dim = 0;
  for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
    for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int i = g_->local_real_offset()[dim]; i < g_->local_offset()[dim]; ++i) {
          EXPECT_EQ(Get(IndexArray(i, j, k)), Get(IndexArray(i+1, j, k))-1);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int i = g_->local_offset()[dim]+g_->local_size()[dim];
             i < g_->local_real_offset()[dim]+g_->local_real_size()[dim]; ++i) {
          EXPECT_EQ(
              Get(IndexArray(i, j, k)), Get(IndexArray(i-1, j, k)) + 1);
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
    DiagonalPeriodicAsymmetry, Grid3DFloatExchangeBoundariesTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-2, -2, -1)),
        ::testing::Values(IndexArray(1, 1, 2)),
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
