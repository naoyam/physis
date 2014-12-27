#include "translator/mpi_opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/mpi_opencl_runtime_builder.h"
#include "translator/reference_runtime_builder.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/mpi_opencl_optimizer.h"
#include "translator/rose_ast_attribute.h"

#include <cstring>
#include <cstdio>
#include <string>

namespace pu = physis::util;
namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

std::string MPIOpenCLTranslator::GetBoundarySuffix(int dim, bool fw) {
  return boundary_suffix_ + "_" +
      toString(dim+1) + "_" + (fw ? "fw" : "bw");
}

std::string MPIOpenCLTranslator::GetBoundarySuffix() {
  return boundary_suffix_;
}

MPIOpenCLTranslator::MPIOpenCLTranslator(const Configuration &config)
    : MPITranslator(config),
      opencl_trans_(new OpenCLTranslator(config)),
      boundary_kernel_width_name_("halo_width"),
      inner_prefix_("_inner"),
      boundary_suffix_("_boundary") {  
  grid_create_name_ = "__PSGridNewMPI";
  target_specific_macro_ = "PHYSIS_MPI_OPENCL";
  flag_multistream_boundary_ = false;
  const pu::LuaValue *lv =
      config.Lookup(Configuration::MULTISTREAM_BOUNDARY);
  if (lv) {
    PSAssert(lv->get(flag_multistream_boundary_));
  }
  if (flag_multistream_boundary_) {
    LOG_INFO() << "Multistream boundary enabled\n";
  }
  validate_ast_ = false;
}

MPIOpenCLTranslator::~MPIOpenCLTranslator() {
  delete opencl_trans_;
}

void MPIOpenCLTranslator::SetUp(SgProject *project,
                                TranslationContext *context) {
  MPITranslator::SetUp(project, context);
  LOG_DEBUG() << "Parent setup done\n";
  opencl_trans_->SetUp(project, context);
  LOG_DEBUG() << "opencl_trans_ setup done\n";
}

void MPIOpenCLTranslator::Finish() {
  LOG_INFO() << "Adding #ifndef " << kernel_mode_macro() << "\n";
  std::string str_insert = "#ifndef ";
  str_insert += kernel_mode_macro();
  si::attachArbitraryText(
      src_->get_globalScope(),
      str_insert,
      PreprocessingInfo::before);
  str_insert = "#endif /* #ifndef ";
  str_insert += kernel_mode_macro();
  str_insert += " */";
  si::attachArbitraryText(
      src_->get_globalScope(),
      str_insert,
      PreprocessingInfo::after);

  add_opencl_extension_pragma();


#ifndef OPENCL_DEVICE_HEADER_PATH
#warning "WARNING: OPENCL_DEVICE_HEADER_PATH not set.\n"
#else
  {
    const char *header_lists[] = {
      "physis/physis_mpi_opencl_device.h",
      NULL
    };

    int num;
    std::string contents_headers = "\n";
    std::string header_path;

#if 0
    const char *cpath_file = __FILE__;
    char *cpath_dup = strdup(cpath_file);
    char *cpos_slash = strrchr(cpath_dup, '/');
    if (cpos_slash)
      *cpos_slash = 0;
    header_path = cpath_dup;
    free(cpath_dup);
    header_path += "/../include/";
#else
    header_path = OPENCL_DEVICE_HEADER_PATH;
    header_path += "/";
#endif

    for (num = 0; header_lists[num]; num++) {
      std::string path = header_path;
      path += header_lists[num];
      const char *file_read = path.c_str();

      FILE *fin = fopen(file_read, "r");
      if (!fin) {
        LOG_DEBUG() << "fopen()ing " << path << " failed\n";
        continue;
      }

#define NUM_BUF_SIZE 1024
      while(1) {
        char cbuf[NUM_BUF_SIZE];
        size_t size = 0;
        size = fread(cbuf, 1, NUM_BUF_SIZE - 1, fin);
        cbuf[size] = 0;
        contents_headers += cbuf;
        if (size < NUM_BUF_SIZE - 1) break;
      }
#undef NUM_BUF_SIZE
      // Insert new line
      contents_headers += "\n";

    } // (num = 0; header_lists[num]; num++)

    LOG_DEBUG() << "Adding the contents of device headers\n";
    str_insert += "\n";
    str_insert += "#ifdef ";
    str_insert += kernel_mode_macro();
    str_insert += "\n";
    str_insert += contents_headers;
    str_insert += "\n#endif /* #ifdef ";
    str_insert += kernel_mode_macro();
    str_insert += " */\n\n";
    str_insert += "#ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "\n";
    si::attachArbitraryText(
        src_->get_globalScope(),
        str_insert,
        PreprocessingInfo::before);
  }
#endif

  // opencl_trans_->Finish();
  MPITranslator::Finish();

}

void MPIOpenCLTranslator::add_opencl_extension_pragma()
{
  // Add pragma for double case
  FOREACH(it, tx_->grid_new_map().begin(), tx_->grid_new_map().end()) {
    Grid *grid = it->second;
    GridType *gt = grid->getType();
    SgType *ty = gt->getElmType();
    if (! isSgTypeDouble(ty)) continue;

    LOG_INFO() << "Adding cl_khr_fp64 pragma for double support";

    std::string str_insert = "";
    str_insert += "#endif /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/\n";

    str_insert += "#ifdef ";
    str_insert += kernel_mode_macro();
    str_insert += "\n";

    str_insert += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    str_insert += "#define PHYSIS_MPI_OPENCL_USE_DOUBLE_STRUCT 1\n";

    str_insert += "\n";
    str_insert += "\n#endif /* #ifdef ";

    str_insert += kernel_mode_macro();
    str_insert += " */\n\n";
    str_insert += "#ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "\n";

    si::attachArbitraryText(
        src_->get_globalScope(),
        str_insert,
        PreprocessingInfo::before);
    break;
  };

  return;

}

} // namespace translator
} // namespace physis
