#include "translator/mpi_openmp_translator.h"

#include "translator/rose_util.h"
#include "translator/translation_context.h"
#include "translator/mpi_runtime_builder.h"

namespace pu = physis::util;
namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

MPIOpenMPTranslator::MPIOpenMPTranslator(const Configuration &config):
    MPITranslator(config)
{
  division_[0] = MPI_OPENMP_DIVISION_X_DEFAULT;
  division_[1] = MPI_OPENMP_DIVISION_Y_DEFAULT;
  division_[2] = MPI_OPENMP_DIVISION_Z_DEFAULT;

  cache_size_[0] = MPI_OPENMP_CACHESIZE_X_DEFAULT;
  cache_size_[1] = MPI_OPENMP_CACHESIZE_Y_DEFAULT;
  cache_size_[2] = MPI_OPENMP_CACHESIZE_Z_DEFAULT;

  target_specific_macro_ = "PHYSIS_MPI_OPENMP";
  grid_create_name_ = "__PSGridNewMPI";

  const pu::LuaValue *lv;
  lv = config.Lookup(Configuration::MPI_OPENMP_DIVISION);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    std::vector<double> v;
    PSAssert(tbl->get(v));
    division_[0] = (int)v[0];
    division_[1] = (int)v[1];
    division_[2] = (int)v[2];
  }
  lv = config.Lookup(Configuration::MPI_OPENMP_CACHESIZE);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    std::vector<double> v;
    PSAssert(tbl->get(v));
    cache_size_[0] = (int)v[0];
    cache_size_[1] = (int)v[1];
    cache_size_[2] = (int)v[2];
  }
} // MPIOpenTranslator

MPIOpenMPTranslator::~MPIOpenMPTranslator() {
} // ~MPIOpenTranslator


} // namespace translator
} // namespace physis

