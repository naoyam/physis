// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/runtime_common.h"
#include "runtime/reduce_cuda.h"
#include "runtime/grid_mpi_cuda_exp.h"
#include "physis/physis_mpi_cuda.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace physis {
namespace runtime {

// This namespace should not need an explict name, but anonymous
// namespace should be just fine. It does not work with nvcc v4.0,
// though. 
namespace reduction_mpi_cuda {

// Adapted from padded_grid_reduction.cu in the thrust examples.

template <typename IndexType, typename ValueType>
struct transform_tuple : 
      public thrust::unary_function< thrust::tuple<IndexType,
                                                   ValueType>, 
                                     thrust::tuple<bool,
                                                   ValueType> > {
  typedef typename thrust::tuple<IndexType,ValueType> InputTuple;
  typedef typename thrust::tuple<bool,ValueType> OutputTuple;

  IndexType n, N;

  transform_tuple(IndexType n, IndexType N) : n(n), N(N) {}

  __host__ __device__
  OutputTuple operator()(const InputTuple& t) const { 
    bool is_valid = (thrust::get<0>(t) % N) < n;
    return OutputTuple(is_valid, thrust::get<1>(t));
  }
};


template <typename IndexType, typename ValueType,
          typename BinaryOperator>
struct reduce_tuple :
    public thrust::binary_function< thrust::tuple<bool,ValueType>,
                                    thrust::tuple<bool,ValueType>,
                                    thrust::tuple<bool,ValueType> >
{
  typedef typename thrust::tuple<bool,ValueType> Tuple;

  __host__ __device__
  Tuple operator()(const Tuple& t0, const Tuple& t1) const { 
    if(thrust::get<0>(t0) && thrust::get<0>(t1)) {// both valid
      return Tuple(true, 
                   BinaryOperator()(thrust::get<1>(t0),
                                    thrust::get<1>(t1)));
    } else if (thrust::get<0>(t0)) {
      return t0;
    } else if (thrust::get<0>(t1)) {
      return t1;
    } else {
      return t1; // if neither is valid then it doesn't matter what we
                 // return
    }
  }
};

//! Reduce a padded grid with binary operation op.
/*
 * \param buf A scalar buffer to store the result.
 * \param op The binary operator to reduce elements.
 * \param dev_grid The address of the grid in the device memory.
 * \param len The number of elements of the grid including padding.
 * \param width The number of elements of the first dimenstion.
 * \param pitch The number of elements of the first dimenstion
 * including padding.
 */
template <class T>
void ReduceGridCUDAPitch(void *buf, PSReduceOp op,
                         void *dev_grid, size_t len,
                         int width, int pitch) {
  typedef typename thrust::tuple<bool, T> result_type;
  T init_val = physis::runtime::GetReductionDefaultValue<T>(op);
  result_type init(true, init_val);
  transform_tuple<size_t, T> unary_op(width, pitch);
  //reduce_tuple<size_t, T> binary_op;
  thrust::device_ptr<T> dev_ptr((T*)dev_grid);
#if 0  
  typedef typename thrust::zip_iterator<thrust::tuple<thrust::counting_iterator<size_t>,
      thrust::device_ptr<T> > > zip_t;
  zip_t zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::counting_iterator<size_t>(0), dev_ptr));
#else
  thrust::zip_iterator<thrust::tuple<thrust::counting_iterator<size_t>,
      thrust::device_ptr<T> > > zip_it =
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::counting_iterator<size_t>(0), dev_ptr));
#endif

  result_type result;
  switch (op)  {
    case PS_MAX:
      result = 
          thrust::transform_reduce(
              zip_it, zip_it + len,
              unary_op, init,
              reduce_tuple<size_t, T, thrust::maximum<T> >());
      break;
    case PS_MIN:
      result = 
          thrust::transform_reduce(
              zip_it, zip_it + len,
              unary_op, init,
              reduce_tuple<size_t, T, thrust::minimum<T> >());
      break;
    case PS_SUM:
      result = 
          thrust::transform_reduce(
              zip_it, zip_it + len,
              unary_op, init,
              reduce_tuple<size_t, T, thrust::plus<T> >());
      break;
    case PS_PROD:
      result = 
          thrust::transform_reduce(
              zip_it, zip_it + len,
              unary_op, init,
              reduce_tuple<size_t, T, thrust::multiplies<T> >());
      break;
    default:
      PSAssert(0);
  }      

  *(T*)buf = thrust::get<1>(result);  

  return;
}

} // namespace reduction_mpi_cuda


template <class T>
int ReduceGridMPICUDA(GridMPICUDAExp *g, PSReduceOp op, T *out) {
  // TODO (Reduction)
#if 0 
  size_t nelms = g->local_size().accumulate(g->num_dims());
  if (nelms == 0) return 0;
  int pitch = g->GetDev()->pitch;
  //LOG_DEBUG() << "Pitch: " << pitch << "\n";
  // ReduceGridCUDA does not handle padding.
  if (pitch == g->local_size()[0]) {
    physis::runtime::ReduceGridCUDA<T>(out, op,
                                       g->_data(), nelms);
  } else {
    IndexArray ls = g->local_size();
    ls[0] = pitch;
    reduction_mpi_cuda::ReduceGridCUDAPitch<T>(
        out, op, g->_data(), ls.accumulate(g->num_dims()),
        g->local_size()[0], pitch);
  }
  return nelms;
#else
  return 0;
#endif  
}

int GridMPICUDAExp::Reduce(PSReduceOp op, void *out) {
  // TODO (Reduction)  
#if 0
  int rv = 0;
  switch (type_) {
    case PS_FLOAT:
      *(float*)out = 0.0f;
      rv = ReduceGridMPICUDA<float>(this, op, (float*)out);
      LOG_DEBUG() << "Reduction: " << *(float*)out << "\n";
      break;
    case PS_DOUBLE:
      rv = ReduceGridMPICUDA<double>(this, op, (double*)out);
      break;
    case PS_INT:
      rv = ReduceGridMPICUDA<int>(this, op, (int*)out);
      break;
    case PS_LONG:
      rv = ReduceGridMPICUDA<long>(this, op, (long*)out);
      break;
    default:
      PSAbort(1);
  }
  return rv;
#else
  return 0;
#endif  
}

} // namespace runtime
} // namespace physis
