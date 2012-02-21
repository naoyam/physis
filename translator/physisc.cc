// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include <boost/program_options.hpp>

#include "translator/config.h"
#include "translator/reference_translator.h"
//#if defined(CUDA_ENABLED)
#include "translator/cuda_translator.h"
//#endif
//#if defined(MPI_ENABLED)
#include "translator/mpi_translator.h"
//#endif
//#if defined(MPI_ENABLED) && defined(CUDA_ENABLED)
#include "translator/mpi_cuda_translator.h"
//#endif
//#if defined(OPENCL_ENABLED)
#include "translator/opencl_translator.h"
//#endif
//#if defined(MPI_ENABLED) && defined(OPENCL_ENABLED)
#include "translator/mpi_opencl_translator.h"
//#endif
//#ifdef defined(MPI_ENABLED) && defined(MPI_OPENMP_ENABLED)
#include "translator/mpi_openmp_translator.h"
//#endif
#include "translator/translator_common.h"
#include "translator/translation_context.h"
#include "translator/translator.h"
#include "translator/configuration.h"

using std::string;
namespace bpo = boost::program_options;
namespace pt = physis::translator;
//namespace pu = physis::util;

namespace physis {
namespace translator {

struct CommandLineOptions {
  bool ref_trans;
  bool cuda_trans;
  bool mpi_trans;
  bool mpi_cuda_trans;
  bool opencl_trans;
  bool mpi_opencl_trans;
  bool mpi_openmp_trans;
  bool mpi_openmp_numa_trans;
  std::pair<bool, string> config_file_path;
  CommandLineOptions(): ref_trans(false), cuda_trans(false),
                        mpi_trans(false), mpi_cuda_trans(false),
                        opencl_trans(false),
                        mpi_opencl_trans(false),
                        mpi_openmp_trans(false),
                        mpi_openmp_numa_trans(false),
                        config_file_path(std::make_pair(false, "")) {}
};

void parseOptions(int argc, char *argv[], CommandLineOptions &opts,
                  vector<string> &rem) {
  // Declare the supported options.
  bpo::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");
  desc.add_options()("ref", "reference translation");
  
  desc.add_options()("config", bpo::value<string>(),
                     "read configuration file");  

  //#ifdef CUDA_ENABLED  
  desc.add_options()("cuda", "CUDA translation");
  //#endif
  //#ifdef MPI_ENABLED
  desc.add_options()("mpi", "MPI translation");
  //#endif
  //#if defined(MPI_ENABLED) && defined(CUDA_ENABLED)
  desc.add_options()("mpi-cuda", "MPI-CUDA translation");
  //#endif
  //#ifdef OPENCL_ENABLED  
  desc.add_options()("opencl", "OPENCL translation");
  //#endif
  //#if defined(MPI_ENABLED) && defined(OPENCL_ENABLED)
  desc.add_options()("mpi-opencl", "MPI-OPENCL translation");
  //#endif
  //#ifdef defined(MPI_ENABLED) && defined(MPI_OPENMP_ENABLED)
  desc.add_options()("mpi-openmp", "MPI-OPENMP translation");
  desc.add_options()("mpi-openmp-numa", "MPI-OPENMP translation");
  //#endif
  desc.add_options()("list-targets", "List available targets");

  bpo::variables_map vm;
  bpo::parsed_options parsed = bpo::command_line_parser(argc, argv).
      options(desc).allow_unregistered().
      style(bpo::command_line_style::default_style &
            ~bpo::command_line_style::allow_guessing).run();
  vector<string> unrec_opts = bpo::collect_unrecognized
      (parsed.options, bpo::include_positional);
  FOREACH (it, unrec_opts.begin(), unrec_opts.end()) {
    rem.push_back(*it);
  }
  bpo::store(parsed, vm);
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(0);
  }

  if (vm.count("config")) {
    opts.config_file_path = make_pair(true, vm["config"].as<string>());
    LOG_DEBUG() << "Configuration file specified: "
                << opts.config_file_path.second << ".\n";    
  }

  if (vm.count("ref")) {
    LOG_DEBUG() << "Reference translation.\n";
    opts.ref_trans = true;
    return;
  }

  //#ifdef CUDA_ENABLED  
  if (vm.count("cuda")) {
    LOG_DEBUG() << "CUDA translation.\n";
    opts.cuda_trans = true;
    return;
  }
  //#endif
  //#ifdef MPI_ENABLED  
  if (vm.count("mpi")) {
    LOG_DEBUG() << "MPI translation.\n";
    opts.mpi_trans = true;
    return;
  }
  //#endif

      //#if defined(MPI_ENABLED) && defined(CUDA_ENABLED)
  if (vm.count("mpi-cuda")) {
    LOG_DEBUG() << "MPI-CUDA translation.\n";
    opts.mpi_cuda_trans = true;
    return;
  }
  //#endif

  //#ifdef OPENCL_ENABLED  
  if (vm.count("opencl")) {
    LOG_DEBUG() << "OPENCL translation.\n";
    opts.opencl_trans = true;
    return;
  }
  //#endif

  //#if defined(MPI_ENABLED) && defined(OPENCL_ENABLED)
  if (vm.count("mpi-opencl")) {
    LOG_DEBUG() << "MPI-OPENCL translation.\n";
    opts.mpi_opencl_trans = true;
    return;
  }
  //#endif

  //#if defined(MPI_ENABLED) && defined(MPI_OPENMP_ENABLED)
  if (vm.count("mpi-openmp")) {
    LOG_DEBUG() << "MPI-OPENMP(-NUMA) translation.\n";
    opts.mpi_openmp_trans = true;
    return;
  }
  if (vm.count("mpi-openmp-numa")) {
    LOG_DEBUG() << "MPI-OPENMP(-NUMA) translation.\n";
    opts.mpi_openmp_numa_trans = true;
    return;
  }
  //#endif

  if (vm.count("list-targets")) {
    StringJoin sj(" ");
    sj << "ref";
    sj << "mpi";
    sj << "cuda";
    sj << "mpi-cuda";
    sj << "opencl";
    sj << "mpi-opencl";
    sj << "mpi-openmp";
    sj << "mpi-openmp-numa";
    std::cout << sj << "\n";
    exit(0);
  }

  LOG_INFO() << "No translation target given.\n";
  std::cout << desc << "\n";
  exit(0);
  
  return;
}

string generate_output_filename(string srcname, string suffix) {
  return srcname.substr(0, srcname.rfind(".")) + "." + suffix;
}

// Set target-wise output file name
void set_output_filename(SgFile *file, string suffix) {
  string name = generate_output_filename(file->get_sourceFileNameWithoutPath(),
                                         suffix);
  file->set_unparse_output_filename(name);
  return;
}

} // namespace translator
} // namespace physis

int main(int argc, char *argv[]) {
  pt::Translator *trans = NULL;
  string filename_suffix;
  string extra_arg;
  pt::CommandLineOptions opts;
  std::vector<std::string> argvec;
  argvec.push_back(argv[0]);
  pt::parseOptions(argc, argv, opts, argvec);

  argvec.push_back("-DPHYSIS_USER");
  
  pt::Configuration config;
  if (opts.config_file_path.first) {
    config.LoadFile(opts.config_file_path.second);
  }
  
  if (opts.ref_trans) {
    trans = new pt::ReferenceTranslator(config);
    filename_suffix = "ref.c";
    argvec.push_back("-DPHYSIS_REF");
  }

  //#ifdef CUDA_ENABLED  
  if (opts.cuda_trans) {
    trans = new pt::CUDATranslator(config);
    filename_suffix = "cuda.cu";
    argvec.push_back("-DPHYSIS_CUDA");
    //    argvec.push_back("-I" + string(CUDA_INCLUDE_DIR));
    //    argvec.push_back("-I" + string(CUDA_CUT_INCLUDE_DIR));
  }
  //#endif

  //#ifdef MPI_ENABLED  
  if (opts.mpi_trans) {
    trans = new pt::MPITranslator(config);
    filename_suffix = "mpi.c";
    argvec.push_back("-DPHYSIS_MPI");
    //argvec.push_back("-I" + string(MPI_INCLUDE_DIR));    
  }
  //#endif

      //#if defined(MPI_ENABLED) && defined(CUDA_ENABLED)
  if (opts.mpi_cuda_trans) {
    trans = new pt::MPICUDATranslator(config);
    filename_suffix = "mpi-cuda.cu";
    argvec.push_back("-DPHYSIS_MPI_CUDA");
    //argvec.push_back("-I" + string(MPI_INCLUDE_DIR));
    //argvec.push_back("-I" + string(CUDA_INCLUDE_DIR));
    //argvec.push_back("-I" + string(CUDA_CUT_INCLUDE_DIR));
  }
  //#endif

  //#ifdef OPENCL_ENABLED  
  if (opts.opencl_trans) {
    trans = new pt::OpenCLTranslator(config);
    filename_suffix = "opencl.c";
    argvec.push_back("-DPHYSIS_OPENCL");
    //    argvec.push_back("-I" + string(OPENCL_INCLUDE_DIR));
  }
  //#endif

  //#if defined(MPI_ENABLED) && defined(OPENCL_ENABLED)  
  if (opts.mpi_opencl_trans) {
    trans = new pt::MPIOpenCLTranslator(config);
    filename_suffix = "mpi-opencl.c";
    argvec.push_back("-DPHYSIS_MPI_OPENCL");
    //    argvec.push_back("-I" + string(OPENCL_INCLUDE_DIR));
  }
  //#endif

  //#if defined(MPI_ENABLED) && defined(MPI_OPENMP_ENABLED)
  if (opts.mpi_openmp_trans) {
    trans = new pt::MPIOpenMPTranslator(config);
    filename_suffix = "mpi-openmp.c";
    argvec.push_back("-DPHYSIS_MPI_OPENMP");
    //argvec.push_back("-I" + string(MPI_INCLUDE_DIR));
    //argvec.push_back("-I" + string(MPI_OPENMP_INCLUDE_DIR));
  }

  if (opts.mpi_openmp_numa_trans) {
    // Use the same MPIOpenMPTranslator
    trans = new pt::MPIOpenMPTranslator(config);
    filename_suffix = "mpi-openmp-numa.c";
    argvec.push_back("-DPHYSIS_MPI_OPENMP");
    argvec.push_back("-DPHYSIS_MPI_OPENMP_NUMA");
    //argvec.push_back("-I" + string(MPI_INCLUDE_DIR));
    //argvec.push_back("-I" + string(MPI_OPENMP_INCLUDE_DIR));
  }
  //#endif

  if (trans == NULL) {
    LOG_INFO() << "No translation done.\n";
    exit(0);
  }

#if defined(PS_VERBOSE_DEBUG)  
  physis::StringJoin sj;
  FOREACH (it, argvec.begin(), argvec.end()) {
    sj << *it;
  }
  LOG_DEBUG() << "Rose command line: " << sj << "\n";
#endif  
  
  // Build the AST used by ROSE
  SgProject* proj = frontend(argvec);
  proj->skipfinalCompileStep(true);

  LOG_INFO() << "AST generation done\n";

  if (proj->numberOfFiles() == 0) {
    LOG_INFO() << "No input source\n";
    exit(0);
  }
  
  // Run internal consistency tests on AST
  //AstTests::runAllTests(proj);
  //generateDOT(*proj, "before");

  pt::TranslationContext tx(proj);

  trans->SetUp(proj, &tx);
  LOG_DEBUG() << "Translating the AST\n";  
  trans->Translate();
  // TODO: optimization is disabled
  //trans->Optimize();
  LOG_DEBUG() << "Translation done\n";
  trans->Finish();
  
  pt::set_output_filename(proj->get_fileList()[0], filename_suffix);
  //SgProject::set_verbose(100000);
  //AstPostProcessing(proj);
  //AstTests::testCompilerGeneratedNodes(proj);
  //generateDOT(*proj, "after");
  //AstTests::runAllTests(proj);

  int b = backend(proj);
  LOG_INFO() << "Code generation complete.\n";
  return b;
}
