// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/grid_mpi_cuda_exp.h"
#include "runtime/grid_space_mpi_cuda.h"
#include "runtime/ipc_mpi.h"
#include "runtime/grid_mpi_debug_util.h"

#include "physis/physis_common.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

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

struct Compound2f {
  float x;
  float y;
};

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

template <class T, int member>
static void InitGrid(GridType *g) {
  if (g->num_dims() == 3) {
    std::cerr << "local_offset: " << g->local_offset()
              << ", local_size: " << g->local_size() << "\n";
    for (int k = 0; k < g->local_size()[2]; ++k) {
      for (int j = 0; j < g->local_size()[1]; ++j) {
        for (int i = 0; i < g->local_size()[0]; ++i) {
          IndexArray ijk = IndexArray(i, j, k);
          IndexArray t = ijk + g->local_offset();
          T v =t[0] + t[1] * g->size()[0]
              + t[2] * g->size()[0]
              * g->size()[1] + member;
          //std::cerr << "write: " << v << "\n";
          GMEMWrite((T*)g->GetAddress(member, t), v);
        }
      }
    }
  } else if  (g->num_dims() == 2) {
    for (int j = 0; j < g->local_size()[1]; ++j) {    
      for (int i = 0; i < g->local_size()[0]; ++i) {
        IndexArray ij = IndexArray(i, j);
        IndexArray t = ij + g->local_offset();
        T v = t[0] + t[1] * g->size()[0] + member;
        GMEMWrite((T*)g->GetAddress(member, t), v);
      }
    }
  } else if  (g->num_dims() == 1) {
    for (int i = 0; i < g->local_size()[0]; ++i) {
      IndexArray t = IndexArray(i+g->local_offset()[0]);
      T v = i + member;
      GMEMWrite((T*)g->GetAddress(member, t), v);
    }
  } else {
    LOG_ERROR() << "Unsupported dimension\n";
    exit(1);
  }
}

template <class T>
class GridUtypeTestBase: public T {
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
    IndexArray stencil_min_m0 = std::tr1::get<2>(this->GetParam());
    IndexArray stencil_max_m0 = std::tr1::get<3>(this->GetParam());    
    IndexArray stencil_min_m1 = std::tr1::get<4>(this->GetParam());
    IndexArray stencil_max_m1 = std::tr1::get<5>(this->GetParam());
    int stencil_min_m[6];
    int stencil_max_m[6];
    for (int i = 0; i < 3; ++i) {
      stencil_min_m[i] = stencil_min_m0[i];
      stencil_min_m[i+3] = stencil_min_m1[i];
      stencil_max_m[i] = stencil_max_m0[i];
      stencil_max_m[i+3] = stencil_max_m1[i];
    }
    width_.fw = stencil_max_;
    width_.bw = stencil_min_ * -1;
    __PSGridTypeMemberInfo minfo[2];
    minfo[0].type = PS_FLOAT;
    minfo[0].rank = 0;
    minfo[0].size = sizeof(float);
    minfo[1].type = PS_FLOAT;
    minfo[1].rank = 0;
    minfo[1].size = sizeof(float);    
    __PSGridTypeInfo tinfo = {PS_USER, sizeof(Compound2f), 2, minfo};
    g_ = gs_->CreateGrid(
        &tinfo, 3, global_size,
        IndexArray(0), stencil_min_, stencil_max_,
        stencil_min_m, stencil_max_m, 0);
    InitGrid<float, 0>(g_);
    InitGrid<float, 1>(g_);    
  }
  
  virtual void TearDown() {
    delete g_;
    delete gs_;
  }

  float Get(const IndexArray &idx) const {
    int member = 0;
    return GMEMRead((float*)g_->GetAddress(member, idx));
  }
  
  GridSpaceMPIType *gs_;
  GridType *g_;
  IndexArray stencil_min_;
  IndexArray stencil_max_;
  Width2 width_;
};

class GridUtypeCopyOutHaloTest:
    public GridUtypeTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, IndexArray, IndexArray,
                 IndexArray, IndexArray, int, bool, bool, int> > > {};

TEST_P(GridUtypeCopyOutHaloTest, CopyOutHalo) {
  int dim = std::tr1::get<6>(GetParam());
  bool diag = std::tr1::get<7>(GetParam());
  bool fw = std::tr1::get<8>(GetParam());
  int member = std::tr1::get<9>(GetParam());
  for (int i = 2; i > dim; --i) {
    gs_->ExchangeBoundaries(g_, member, i, width_, false, false);    
  }
  g_->CopyoutHalo(dim, width_, fw, diag);
  LOG_DEBUG() << "Copyout done\n";
  if (g_->halo(member)(dim, fw) > 0 &&
      ((fw && gs_->my_idx()[dim] != 0) ||
       (!fw && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1))) {
    float *buf = (float*)g_->GetHaloSelfHost(dim, fw, member)->Get();
    IndexArray idx_begin(0);
    IndexArray idx_end = g_->local_real_size(member);
    if (fw) {
      idx_begin[dim] = g_->halo(member)(dim, false);
    } else {
      idx_begin[dim] = g_->local_real_size(member)[dim] -
          g_->halo(member)(dim, true) - g_->halo(member)(dim, false);
    }
    idx_end[dim] = idx_begin[dim] + g_->halo(member)(dim, fw);
    int buf_idx = 0;
    IndexArray halo_size = g_->local_real_size(member);
    halo_size[dim] = g_->halo(member)(dim, fw);
    for (int k = idx_begin[2]; k < idx_end[2]; ++k) {
      if (dim < 2) {
        // handle edge processe, which do not have halo unless
        // periodic
        if (k < (int)g_->halo(member)(2, false)) {
          if (gs_->my_idx()[2] == 0) {
            buf_idx += halo_size[0] * halo_size[1];
            continue;
          }
        } else if (k >= (int)g_->halo(member)(2, false) + g_->local_size()[2]) {
          if (gs_->my_idx()[2] == gs_->proc_size()[2] - 1) {
            break;
          }
        }
      }
      for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
        if (dim > 1) {
          if (j < (int)g_->halo(member)(1, false) ||
              j >= (int)g_->halo(member)(1, false) + g_->local_size()[1]) {
            buf_idx += g_->local_real_size(member)[0];
            continue;
          }
        } else if (dim < 1) {
          if ((j < (int)g_->halo(member)(1, false) &&
              gs_->my_idx()[1] == 0) ||
              (j >= (int)g_->halo(member)(1, false) + g_->local_size()[1] &&
               gs_->my_idx()[1] == gs_->proc_size()[1] - 1)) {
            buf_idx += halo_size[0];
            continue;
          }
        }
        for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
          if (dim > 0) {
            if (i < (int)g_->halo(member)(0, false) ||
                i >= (int)g_->halo(member)(0, false) + g_->local_size()[0]) {
              buf_idx += 1;
              continue;
            }
          }
          
          IndexArray ijk = IndexArray(i, j, k) + g_->local_real_offset(member);
          float v = ijk[0] + ijk[1] * g_->size()[0]
              + ijk[2] * g_->size()[0] * g_->size()[1] + member;
          ASSERT_EQ(buf[buf_idx], v)
              << "[" << ipc->GetRank() << "]"
              << "(i,j,k) = (" << i << "," << j << "," << k << "), "
              << "buf idx: " << buf_idx
              << ", ijk: " << ijk
              << ", local_real_offset: " << g_->local_real_offset(member)
              << ", local_real_size: " << g_->local_real_size(member)
              << ", fw: " << fw
              << ", dim: " << dim;
              //              << ", next: " << buf[buf_idx+1];
          ++buf_idx;
#if 0          
          std::cout 
              << "PASS: (i,j,k) = (" << i << "," << j << "," << k << "), "
              << "v: " << v << "\n";
#endif          
        }
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(
    EachDirectionNoDiag7pt, GridUtypeCopyOutHaloTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -1, -1)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Values(IndexArray(-1,-1,-1)),
        ::testing::Values(IndexArray(1,1,1)),
        ::testing::Values(IndexArray(-1,-1,-1)),
        ::testing::Values(IndexArray(1,1,1)),
        ::testing::Range(2, 3),
        ::testing::Values(false),
        ::testing::Bool(),
        ::testing::Values(0, 1)));
#if 0
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

#endif

class GridUtypeExchangeBoundariesDimTest:
    public GridUtypeTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, IndexArray, IndexArray,
                 IndexArray, IndexArray, int, int> > > {};


TEST_P(GridUtypeExchangeBoundariesDimTest, ExchangeBoundaries) {
  //g_->Print(std::cout);
  //std::cout << "\n";
  int dim = std::tr1::get<6>(GetParam());  
  int member = std::tr1::get<7>(GetParam());  

  for (int i = 2; i >= dim; --i) {
    gs_->ExchangeBoundaries(g_, 0, i, width_, false, false);
  }
  IndexArray idx_begin = g_->local_offset();
  IndexArray idx_end = g_->local_offset() + g_->local_size();
  float diff = 1;
  for (int i = 0; i < dim; ++i) {
    diff *= N;
  }
  idx_begin[dim] = g_->local_real_offset(member)[dim];
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
  idx_end[dim] = g_->local_real_offset(member)[dim]+g_->local_real_size(member)[dim];
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
    EachDimension7pt, GridUtypeExchangeBoundariesDimTest,
    ::testing::Combine(
        ::testing::Values(IndexArray(-1, -1, -1)),
        ::testing::Values(IndexArray(1, 1, 1)),
        ::testing::Values(IndexArray(-1,-1,-1)),
        ::testing::Values(IndexArray(1,1,1)),
        ::testing::Values(IndexArray(-1,-1,-1)),
        ::testing::Values(IndexArray(1,1,1)),
        ::testing::Range(2, 3),
        ::testing::Values(0, 1)));

#if 0
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
#endif
#if 0
class Grid3DFloatExchangeBoundariesTest:
    public Grid3DFloatTestBase< ::testing::TestWithParam<
      tr1::tuple<IndexArray, IndexArray, bool, bool> > > {};

TEST_P(Grid3DFloatExchangeBoundariesTest, ExchangeBoundaries) {
  int member = 0;  
  bool diag = std::tr1::get<2>(GetParam());
  bool periodic = std::tr1::get<3>(GetParam());
  gs_->ExchangeBoundaries(g_, 0, 2, width_, diag, periodic);
  gs_->ExchangeBoundaries(g_, 0, 1, width_, diag, periodic);
  gs_->ExchangeBoundaries(g_, 0, 0, width_, diag, periodic);
  int dim = 2;
  IndexArray idx_begin = g_->local_real_offset(member);
  IndexArray idx_end = g_->local_real_offset(member) + g_->local_real_size(member);
  if (!periodic) {
    for (int i = 0; i < 3; ++i) {
      if (gs_->my_idx()[i] == 0) {
        idx_begin[i] +=  g_->halo(member)(i, false);
      } else if (gs_->my_idx()[i] == gs_->proc_size()[i]-1) {
        idx_end[i] -= g_->halo(member)(i, true);
      }
    }
  }
  for (int j = idx_begin[1]; j < idx_end[1]; ++j) {
    for (int i = idx_begin[0]; i < idx_end[0]; ++i) {
      if (!periodic && gs_->my_idx()[dim] != 0) {
        for (int k = g_->local_real_offset(member)[2]; k < g_->local_offset()[2]; ++k) {
          ASSERT_EQ(
              Get(IndexArray(i, j, k)),
              Get(IndexArray(i, j, k+1)) - N*N);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int k = g_->local_offset()[2]+g_->local_size()[2];
             k < g_->local_real_offset(member)[2]+g_->local_real_size(member)[2]; ++k) {
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
        for (int j = g_->local_real_offset(member)[dim];
             j < g_->local_offset()[dim]; ++j) {
          ASSERT_EQ(Get(IndexArray(i, j, k)),
                    Get(IndexArray(i, j+1, k)) - N);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int j = g_->local_offset()[dim]+g_->local_size()[dim];
             j < g_->local_real_offset(member)[dim]+g_->local_real_size(member)[dim]; ++j) {
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
        for (int i = g_->local_real_offset(member)[dim]; i < g_->local_offset()[dim]; ++i) {
          ASSERT_EQ(Get(IndexArray(i, j, k)),
                    Get(IndexArray(i+1, j, k))-1);
        }
      }
      if (!periodic && gs_->my_idx()[dim] != gs_->proc_size()[dim]-1) {
        for (int i = g_->local_offset()[dim]+g_->local_size()[dim];
             i < g_->local_real_offset(member)[dim]+g_->local_real_size(member)[dim]; ++i) {
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

#endif
int main(int argc, char *argv[]) {
  ::testing::InitGoogleMock(&argc, argv);
  ipc = InterProcCommMPI::GetInstance();
  ipc->Init(&argc, &argv);
  int proc_num_dims = GetProcessDim(&argc, &argv, proc_size);
  if (proc_num_dims == -1) {
    // if no process size specified
    proc_size = IntArray(1, 1, ipc->GetNumProcs());
  }
  //int i = 0;
  int i = 1;
  while (i == 0) {
    sleep(10);
  }
  int x = RUN_ALL_TESTS();
  MPI_Finalize();
  return x;
}
