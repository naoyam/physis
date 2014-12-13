#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "runtime/buffer_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace ::testing;
using namespace ::std;

namespace physis {
namespace runtime {


TEST(BufferCUDAHost, EnsureCapacity) {
  BufferCUDAHost buf;
  size_t s = 100;
  buf.EnsureCapacity(s);
  ASSERT_EQ(buf.size(), s);
}

TEST(BufferCUDADev, EnsureCapacity) {
  BufferCUDADev buf;
  size_t s = 100;
  buf.EnsureCapacity(s);
  ASSERT_EQ(buf.size(), s);
}

TEST(BufferCUDADev, Copyout2D) {
  BufferCUDADev buf;
  int s = 4*4;
  buf.EnsureCapacity(s*sizeof(int));
  int *p = (int*)buf.Get();
  int h[s];
  for (int i = 0; i < s; ++i) {
    h[i] = i;
  }
  cudaMemcpy(p, h, sizeof(int)*s, cudaMemcpyHostToDevice);
  int q[4];
  LOG_DEBUG() << "4, 1\n";
  buf.Copyout(sizeof(int), 2, IndexArray(4,4),
              q, IndexArray(0,0), IndexArray(4,1));
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(q[i], i);
  }
  LOG_DEBUG() << "1, 4\n";
  buf.Copyout(sizeof(int), 2, IndexArray(4,4),
              q, IndexArray(1,0), IndexArray(1,4));
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(q[i], i*4+1);
  }
  LOG_DEBUG() << "3, 1\n";
  buf.Copyout(sizeof(int), 2, IndexArray(4,4),
              q, IndexArray(0,0), IndexArray(3,1));
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(q[i], i);
  }
  buf.Copyout(sizeof(int), 2, IndexArray(4,4),
              q, IndexArray(1,1), IndexArray(2,2));
  int k = 0;
  for (int i = 1; i < 3; ++i) {
    for (int j = 1; j < 3; ++j) {
      ASSERT_EQ(j+i*4, q[k]);
      ++k;
    }
  }
}

TEST(BufferCUDADev, Copyin2D) {
  BufferCUDADev buf;
  int s = 4*4;
  buf.EnsureCapacity(s*sizeof(int));
  int p[s];
  int q[s];  
  for (int i = 0; i < s; ++i) {
    p[i] = 0;
  }
  buf.CopyinAll(p);
  for (int i = 0; i < s; ++i) {
    p[i] = i;
  }
  buf.Copyin(sizeof(int), 2, IndexArray(4,4),
             p, IndexArray(0,0), IndexArray(4,1));
  buf.CopyoutAll(q);
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(q[i], i);
  }
  buf.Copyin(sizeof(int), 2, IndexArray(4,4),
             p, IndexArray(1,0), IndexArray(1,4));
  buf.CopyoutAll(q);  
  for (int j = 0; j < 4; ++j) {
    for (int i = 1; i < 2; ++i) {
      ASSERT_EQ(q[i+j*4], i-1+j);
    }
  }
  buf.Copyin(sizeof(int), 2, IndexArray(4,4),
             p, IndexArray(1,1), IndexArray(2,2));
  buf.CopyoutAll(q);
  for (int j = 1; j < 3; ++j) {
    for (int i = 1; i < 3; ++i) {
      ASSERT_EQ(q[i+j*4], i - 1 + 2 * (j-1));
    }
  }
}


TEST(BufferCUDADev, Copyout3D) {
  BufferCUDADev buf;
  int rank = 3;
  IndexArray size(3, 4, 5);
  int ne = size.accumulate(rank);
  buf.EnsureCapacity(ne*sizeof(int));
  int *p = (int*)buf.Get();
  int h[ne];
  for (int i = 0; i < ne; ++i) {
    h[i] = i;
  }
  cudaMemcpy(p, h, sizeof(int)*ne, cudaMemcpyHostToDevice);
  int q[ne];
  buf.Copyout(sizeof(int), rank, size,
              q, IndexArray(0,0,0), IndexArray(3, 1 ,1));
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(q[i], i);
  }
  buf.Copyout(sizeof(int), rank, size,
              q, IndexArray(0,0,1), IndexArray(3, 4 ,1));
  for (int i = 0; i < 3*4; ++i) {
    ASSERT_EQ(q[i], i+3*4);
  }

  buf.Copyout(sizeof(int), rank, size,
              q, IndexArray(1,0,0), IndexArray(1, 4 ,1));
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(q[i], 1+i*3);
  }

  buf.Copyout(sizeof(int), rank, size,
              q, IndexArray(0,0,0), IndexArray(1, 4 ,5));
  int k = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      ASSERT_EQ(q[k], j*3+i*3*4);
      ++k;
    }
  }
  buf.Copyout(sizeof(int), rank, size,
              q, IndexArray(1,0,0), IndexArray(2, 4 ,5));
  k = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int l = 1; l < 3; ++l) {
        ASSERT_EQ(q[k], l+j*3+i*3*4);
        ++k;
      }
    }
  }
}

TEST(BufferCUDADev, Copyin3D) {
  BufferCUDADev buf;
  int rank = 3;
  IndexArray size(3, 4, 5);
  int ne = size.accumulate(rank);
  buf.EnsureCapacity(ne*sizeof(int));
  int p[ne];
  int q[ne];  
  for (int i = 0; i < ne; ++i) {
    p[i] = 0;
  }
  buf.CopyinAll(p);
  for (int i = 0; i < ne; ++i) {
    p[i] = i;
  }
  buf.Copyin(sizeof(int), rank, size,
             p, IndexArray(0,0,0), IndexArray(3, 1 ,1));
  buf.CopyoutAll(q);  
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(q[i], i);
  }
  buf.Copyin(sizeof(int), rank, size,
             p, IndexArray(1,0,0), IndexArray(1, 4 ,1));
  buf.CopyoutAll(q);    
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(i, q[1+i*3]);
  }
  buf.Copyin(sizeof(int), rank, size,
             p, IndexArray(0,0,0), IndexArray(1, 4 ,5));
  buf.CopyoutAll(q);    
  int k = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      ASSERT_EQ(k, q[j*3+i*3*4]);
      ++k;
    }
  }
  
  buf.Copyin(sizeof(int), rank, size,
             p, IndexArray(0,0,1), IndexArray(3, 4 ,1));
  buf.CopyoutAll(q);    
  k = 0;
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 3; ++i) {
      ASSERT_EQ(k, q[i+j*3+3*4]);
      ++k;
    }
  }
  
  buf.Copyin(sizeof(int), rank, size,
             p, IndexArray(1,0,0), IndexArray(2, 4 ,5));
  buf.CopyoutAll(q);      
  k = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int l = 1; l < 3; ++l) {
        ASSERT_EQ(k, q[l+j*3+i*3*4]);
        ++k;
      }
    }
  }
}

} // namespace runtime
} // namespace physis

  
int main(int argc, char *argv[]) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
