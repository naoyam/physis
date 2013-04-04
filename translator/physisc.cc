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
#include "translator/cuda_translator.h"
#include "translator/mpi_translator.h"
//#include "translator/mpi_translator2.h"
#include "translator/mpi_cuda_translator.h"
#include "translator/opencl_translator.h"
#include "translator/mpi_opencl_translator.h"
#include "translator/mpi_openmp_translator.h"
#include "translator/translator_common.h"
#include "translator/translation_context.h"
#include "translator/translator.h"
#include "translator/runtime_builder.h"
#include "translator/cuda_runtime_builder.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/mpi_cuda_runtime_builder.h"
#include "translator/configuration.h"
#include "translator/optimizer/optimizer.h"
#include "translator/optimizer/reference_optimizer.h"
#include "translator/optimizer/cuda_optimizer.h"
#include "translator/optimizer/mpi_optimizer.h"
#include "translator/optimizer/mpi_cuda_optimizer.h"

using std::string;
namespace bpo = boost::program_options;
namespace pt = physis::translator;
namespace pto = physis::translator::optimizer;

namespace physis {
namespace translator {

struct CommandLineOptions {
  bool ref_trans;
  bool cuda_trans;
  bool mpi_trans;
  //bool mpi2_trans;  
  bool mpi_cuda_trans;
  bool opencl_trans;
  bool mpi_opencl_trans;
  bool mpi_openmp_trans;
  bool mpi_openmp_numa_trans;
  std::pair<bool, string> config_file_path;
  CommandLineOptions(): ref_trans(false), cuda_trans(false),
                        mpi_trans(false),
                        //mpi2_trans(false),
                        mpi_cuda_trans(false),
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
  desc.add_options()("cuda", "CUDA translation");
  desc.add_options()("mpi", "MPI translation");
  //desc.add_options()("mpi2", "MPI translation v2");
  desc.add_options()("mpi-cuda", "MPI-CUDA translation");
  desc.add_options()("opencl", "*EXPERIMENTAL* OpenCL translation");
  desc.add_options()("mpi-opencl", "*EXPERIMENTAL* MPI-OpenCL translation");
  desc.add_options()("mpi-openmp", "*EXPERIMENTAL* MPI-OpenMP translation");
  desc.add_options()("mpi-openmp-numa", "*EXPERIMENTAL* NUMA-aware MPI-OpenMP translation");
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

  if (vm.count("cuda")) {
    LOG_DEBUG() << "CUDA translation.\n";
    opts.cuda_trans = true;
    return;
  }

  if (vm.count("mpi")) {
    LOG_DEBUG() << "MPI translation.\n";
    opts.mpi_trans = true;
    return;
  }

  // if (vm.count("mpi2")) {
  //   LOG_DEBUG() << "MPI translation v2.\n";
  //   opts.mpi2_trans = true;
  //   return;
  // }

  if (vm.count("mpi-cuda")) {
    LOG_DEBUG() << "MPI-CUDA translation.\n";
    opts.mpi_cuda_trans = true;
    return;
  }

  if (vm.count("opencl")) {
    LOG_DEBUG() << "OpenCL translation.\n";
    LOG_WARNING() << "OpenCL: EXPERIMENTAL FEATURE\n";
    opts.opencl_trans = true;
    return;
  }

  if (vm.count("mpi-opencl")) {
    LOG_DEBUG() << "MPI-OpenCL translation.\n";
    LOG_WARNING() << "MPI-OpenCL: EXPERIMENTAL FEATURE\n";
    opts.mpi_opencl_trans = true;
    return;
  }

  if (vm.count("mpi-openmp")) {
    LOG_DEBUG() << "MPI-OpenMP translation.\n";
    LOG_WARNING() << "MPI-OpenMP: EXPERIMENTAL FEATURE\n";
    opts.mpi_openmp_trans = true;
    return;
  }

  if (vm.count("mpi-openmp-numa")) {
    LOG_DEBUG() << "MPI-OpenMP-NUMA translation.\n";
    LOG_WARNING() << "MPI-OpenMP-NUMA: EXPERIMENTAL FEATURE\n";
    opts.mpi_openmp_numa_trans = true;
    return;
  }

  if (vm.count("list-targets")) {
    StringJoin sj(" ");
    sj << "ref";
    sj << "mpi";
    //    sj << "mpi2";    
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

static pt::RuntimeBuilder *GetRTBuilder(SgProject *proj,
                                        CommandLineOptions &opts) {
  pt::RuntimeBuilder *builder = NULL;
  SgScopeStatement *gs = si::getFirstGlobalScope(proj);
  if (opts.ref_trans) {
    builder = new pt::ReferenceRuntimeBuilder(gs);
  } else if (opts.cuda_trans) {
    builder = new pt::CUDARuntimeBuilder(gs);
  } else if (opts.mpi_trans) {
    builder = new pt::MPIRuntimeBuilder(gs);
  // } else if (opts.mpi2_trans) {
  //   builder = new pt::MPIRuntimeBuilder(gs);
  } else if (opts.mpi_cuda_trans) {
    builder = new pt::MPICUDARuntimeBuilder(gs);
  }
  if (builder == NULL) {
    LOG_WARNING() << "No runtime builder found for this target\n";
  }
  return builder;
}
static pto::Optimizer *GetOptimizer(TranslationContext *tx,
                                    SgProject *proj,
                                    RuntimeBuilder *builder,
                                    CommandLineOptions &opts,
                                    Configuration *cfg) {
  pto::Optimizer *optimizer = NULL;
  if (opts.ref_trans) {
    optimizer = new pto::ReferenceOptimizer(proj, tx,
                                            builder, cfg);
  } else if (opts.cuda_trans) {
    optimizer = new pto::CUDAOptimizer(proj, tx,
                                       builder, cfg);
  } else if (opts.mpi_trans) {
    optimizer = new pto::MPIOptimizer(proj, tx,
                                      builder, cfg);    
  } else if (opts.mpi_cuda_trans) {
    optimizer = new pto::MPICUDAOptimizer(proj, tx,
                                          builder, cfg);        
  }
  if (optimizer == NULL) {
    LOG_WARNING() << "No optimizer found for this target\n";
  }
  return optimizer;
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

  if (opts.cuda_trans) {
    trans = new pt::CUDATranslator(config);
    filename_suffix = "cuda.cu";
    argvec.push_back("-DPHYSIS_CUDA");
  }

  if (opts.mpi_trans) {
    trans = new pt::MPITranslator(config);
    filename_suffix = "mpi.c";
    argvec.push_back("-DPHYSIS_MPI");
  }

  // if (opts.mpi2_trans) {
  //   trans = new pt::MPITranslator2(config);
  //   filename_suffix = "mpi2.c";
  //   argvec.push_back("-DPHYSIS_MPI");
  // }

  if (opts.mpi_cuda_trans) {
    trans = new pt::MPICUDATranslator(config);
    filename_suffix = "mpi-cuda.cu";
    argvec.push_back("-DPHYSIS_MPI_CUDA");
  }

  if (opts.opencl_trans) {
    trans = new pt::OpenCLTranslator(config);
    filename_suffix = "opencl.c";
    argvec.push_back("-DPHYSIS_OPENCL");
  }

  if (opts.mpi_opencl_trans) {
    trans = new pt::MPIOpenCLTranslator(config);
    filename_suffix = "mpi-opencl.c";
    argvec.push_back("-DPHYSIS_MPI_OPENCL");
  }

  if (opts.mpi_openmp_trans) {
    trans = new pt::MPIOpenMPTranslator(config);
    filename_suffix = "mpi-openmp.c";
    argvec.push_back("-DPHYSIS_MPI_OPENMP");
  }

  if (opts.mpi_openmp_numa_trans) {
    // Use the same MPIOpenMPTranslator
    trans = new pt::MPIOpenMPTranslator(config);
    filename_suffix = "mpi-openmp-numa.c";
    argvec.push_back("-DPHYSIS_MPI_OPENMP");
    argvec.push_back("-DPHYSIS_MPI_OPENMP_NUMA");
  }

  if (trans == NULL) {
    LOG_INFO() << "No translation done.\n";
    exit(0);
  }

#if defined(PS_VERBOSE)  
  physis::StringJoin sj;
  FOREACH (it, argvec.begin(), argvec.end()) {
    sj << *it;
  }
  LOG_VERBOSE() << "Rose command line: " << sj << "\n";
#endif  
  
  // Build AST
  SgProject* proj = frontend(argvec);
  proj->skipfinalCompileStep(true);

  LOG_INFO() << "AST generation done\n";

  if (proj->numberOfFiles() == 0) {
    LOG_INFO() << "No input source\n";
    exit(0);
  }

  // Run internal consistency tests on AST  
  LOG_INFO() << "Checking AST consistency.\n";
  AstTests::runAllTests(proj);
  LOG_INFO() << "AST validated successfully.\n";

  pt::TranslationContext tx(proj);

  trans->SetUp(proj, &tx);

  // TODO: reuse this builder with the translator
  pt::RuntimeBuilder *rt_builder = GetRTBuilder(proj, opts);
  pto::Optimizer *optimizer =
      GetOptimizer(&tx, proj, rt_builder, opts, &config);

  LOG_INFO() << "Performing optimization Stage 1\n";
  // optimizer is null if not defined for this target
  if (optimizer) optimizer->Stage1();
  LOG_INFO() << "Optimization Stage 1 done\n";
  
  LOG_INFO() << "Translating the AST\n";
  trans->Translate();
  // TODO: optimization is disabled
  //trans->Optimize();
  LOG_INFO() << "Translation done\n";
  
  LOG_INFO() << "Performing optimization Stage 2\n";
  if (optimizer) optimizer->Stage2();
  LOG_INFO() << "Optimization Stage 2 done\n";
  
  trans->Finish();
  delete optimizer;
  
  pt::set_output_filename(proj->get_fileList()[0], filename_suffix);

  int b = backend(proj);
  LOG_INFO() << "Code generation complete.\n";
  return b;
}
