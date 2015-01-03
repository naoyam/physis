// Licensed under the BSD license. See LICENSE.txt for more details.

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#include "translator/config.h"
#include "translator/reference_translator.h"
#include "translator/cuda_translator.h"
#include "translator/mpi_translator.h"
//#include "translator/mpi_translator2.h"
#include "translator/translator_common.h"
#include "translator/translation_context.h"
#include "translator/translator.h"
#include "translator/builder_interface.h"
#include "translator/cuda_runtime_builder.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/configuration.h"
#include "translator/optimizer/optimizer.h"
#include "translator/optimizer/reference_optimizer.h"
#include "translator/optimizer/cuda_optimizer.h"
#include "translator/optimizer/mpi_optimizer.h"
#ifdef MPI_CUDA_TRANSLATOR_ENABLED
#include "translator/mpi_cuda_translator.h"
#include "translator/mpi_cuda_runtime_builder.h"
#include "translator/optimizer/mpi_cuda_optimizer.h"
#endif
#ifdef OPENCL_TRANSLATOR_ENABLED
#include "translator/opencl_translator.h"
#endif
#ifdef MPI_OPENCL_TRANSLATOR_ENABLED
#include "translator/mpi_opencl_translator.h"
#endif
#ifdef MPI_OPENMP_TRANSLATOR_ENABLED
#include "translator/mpi_openmp_translator.h"
#endif
#ifdef CUDA_HM_TRANSLATOR_ENABLED
#include "translator/cuda_hm_translator.h"
#include "translator/cuda_hm_runtime_builder.h"
#endif
#include "translator/fortran_output_fix.h"

using std::string;
namespace bpo = boost::program_options;
namespace pt = physis::translator;
namespace pto = physis::translator::optimizer;
namespace ru = physis::translator::rose_util;

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
  bool cuda_hm_trans;
  std::pair<bool, string> config_file_path;
  CommandLineOptions(): ref_trans(false), cuda_trans(false),
                        mpi_trans(false),
                        //mpi2_trans(false),
                        mpi_cuda_trans(false),
                        opencl_trans(false),
                        mpi_opencl_trans(false),
                        mpi_openmp_trans(false),
                        mpi_openmp_numa_trans(false),
                        cuda_hm_trans(false),
                        config_file_path(std::make_pair(false, "")) {}
};

void parseOptions(int argc, char *argv[], CommandLineOptions &opts,
                  vector<string> &rem) {
  // Declare the supported options.
  bpo::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message");
  desc.add_options()("config", bpo::value<string>(),
                     "Read configuration file");
  desc.add_options()("ref", "Reference translation");
#ifdef CUDA_TRANSLATOR_ENABLED  
  desc.add_options()("cuda", "CUDA translation");
#endif  
#ifdef CUDA_HM_TRANSLATOR_ENABLED  
  desc.add_options()(
      "cuda-hm", "*EXPERIMENTAL* CUDA with host memory translation");
#endif
#ifdef MPI_TRANSLATOR_ENABLED
  desc.add_options()("mpi", "MPI translation");
#endif  
  //desc.add_options()("mpi2", "MPI translation v2");
#ifdef MPI_CUDA_TRANSLATOR_ENABLED  
  desc.add_options()("mpi-cuda", "MPI-CUDA translation");
#endif
#ifdef OPENCL_TRANSLATOR_ENABLED
  desc.add_options()("opencl", "*EXPERIMENTAL* OpenCL translation");
#endif
#ifdef MPI_OPENCL_TRANSLATOR_ENABLED
  desc.add_options()("mpi-opencl", "*EXPERIMENTAL* MPI-OpenCL translation");
#endif
#ifdef MPI_OPENMP_TRANSLATOR_ENABLED  
  desc.add_options()("mpi-openmp", "*EXPERIMENTAL* MPI-OpenMP translation");
  desc.add_options()(
      "mpi-openmp-numa", "*EXPERIMENTAL* NUMA-aware MPI-OpenMP translation");
#endif  
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

  if (vm.count("cuda-hm")) {
    LOG_DEBUG() << "CUDA with host memory translation.\n";
    opts.cuda_hm_trans = true;
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
#ifdef CUDA_HM_TRANSLATOR_ENABLED
    sj << "cuda-hm";
#endif
#ifdef MPI_CUDA_TRANSLATOR_ENABLED    
    sj << "mpi-cuda";
#endif
#ifdef OPENCL_TRANSLATOR_ENABLED
    sj << "opencl";
#endif
#ifdef MPI_OPENCL_TRANSLATOR_ENABLED
    sj << "mpi-opencl";
#endif
#ifdef MPI_OPENMP_TRANSLATOR_ENABLED
    sj << "mpi-openmp";
    sj << "mpi-openmp-numa";
#endif
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
  LOG_INFO() << "Output file name: " << name << "\n";
  file->set_unparse_output_filename(name);
  return;
}

static pt::BuilderInterface *GetRTBuilder(SgProject *proj,
                                          const Configuration &config,
                                          CommandLineOptions &opts) {
  pt::BuilderInterface *builder = NULL;
  SgScopeStatement *gs = si::getFirstGlobalScope(proj);
  if (opts.ref_trans) {
    builder = new pt::ReferenceRuntimeBuilder(gs, config);
  } else if (opts.cuda_trans) {
    builder = new pt::CUDARuntimeBuilder(gs, config);
#ifdef CUDA_HM_TRANSLATOR_ENABLED    
  } else if (opts.cuda_hm_trans) {
    builder = new pt::CUDAHMRuntimeBuilder(gs, config);
#endif    
  } else if (opts.mpi_trans) {
    builder = new pt::MPIRuntimeBuilder(gs, config);
  // } else if (opts.mpi2_trans) {
  //   builder = new pt::MPIRuntimeBuilder(gs);
#ifdef MPI_CUDA_TRANSLATOR_ENABLED    
  } else if (opts.mpi_cuda_trans) {
    builder = new pt::MPICUDARuntimeBuilder(gs, config);
#endif    
  }
  if (builder == NULL) {
    LOG_WARNING() << "No runtime builder found for this target\n";
  }
  return builder;
}
static pto::Optimizer *GetOptimizer(TranslationContext *tx,
                                    SgProject *proj,
                                    BuilderInterface *builder,
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
#ifdef MPI_CUDA_TRANSLATOR_ENABLED    
  } else if (opts.mpi_cuda_trans) {
    optimizer = new pto::MPICUDAOptimizer(proj, tx,
                                          builder, cfg);
#endif    
  }
  if (optimizer == NULL) {
    LOG_WARNING() << "No optimizer found for this target\n";
  }
  return optimizer;
}
/** get all kernel functions from proj.
 * @param[in] proj
 * @return    kernel functions
 */
static std::vector<SgFunctionDeclaration *> GetRunKernelFunc(SgProject *proj) {
  std::vector<SgFunctionDeclaration *> kernel_funcs;
  Rose_STL_Container<SgNode *> funcs =
      NodeQuery::querySubTree(proj, V_SgFunctionDeclaration);
  FOREACH(it, funcs.begin(), funcs.end()) {
    SgFunctionDeclaration *func = isSgFunctionDeclaration(*it);
    RunKernelAttribute *run_kernel_attr =
        rose_util::GetASTAttribute<RunKernelAttribute>(func);
    if (run_kernel_attr) {
      kernel_funcs.push_back(run_kernel_attr->stencil_map()->getKernel());
      //LOG_DEBUG() << "kernel_funcs.push_back: " << run_kernel_attr->stencil_map()->getKernel()->get_name() << "\n";
      kernel_funcs.push_back(func);
      //LOG_DEBUG() << "kernel_funcs.push_back: " << func->get_name() << "\n";
    }
    if (rose_util::GetASTAttribute<RunKernelCallerAttribute>(func)) {
      kernel_funcs.push_back(func);
      //LOG_DEBUG() << "kernel_funcs.push_back: " << func->get_name() << "\n";
      SgVariableDeclaration *func_pointer =
          sb::buildVariableDeclaration(
              func->get_name(),
              sb::buildPointerType(func->get_type()));
      si::replaceStatement(func, func_pointer);
    }
  }
  return kernel_funcs;
}
/** replace original kernel functions,
 *  with typedef, enum and struct declarations.
 * @param[in] proj
 * @param[in] kernel_funcs ... original kernel functions
 */
static void ReplaceCloneRunKernelFunc(
    SgProject *proj, std::vector<SgFunctionDeclaration *> &kernel_funcs) {
  SgScopeStatement *sc = si::getFirstGlobalScope(proj);
  std::vector<SgStatement *> save;
  /* save typedef declarations */
  SgNodePtrList typedef_decls =
      NodeQuery::querySubTree(proj, V_SgTypedefDeclaration);
  FOREACH(it, typedef_decls.begin(), typedef_decls.end()) {
    save.push_back(isSgTypedefDeclaration(*it));
  }
  /* save enum declarations */
  SgNodePtrList enum_decls =
      NodeQuery::querySubTree(proj, V_SgEnumDeclaration);
  FOREACH(it, enum_decls.begin(), enum_decls.end()) {
    save.push_back(isSgEnumDeclaration(*it));
  }
  /* save struct declarations */
  SgNodePtrList struct_decls =
      NodeQuery::querySubTree(proj, V_SgClassDeclaration);
  FOREACH(it, struct_decls.begin(), struct_decls.end()) {
    SgClassDeclaration *class_decl = isSgClassDeclaration(*it);
    if (si::isStructType(class_decl->get_type())) {
      save.push_back(class_decl);
    }
  }
  /* remove statement (without #include) */
  SgStatement *s;
  while ((s = si::getLastStatement(sc))) {
    si::removeStatement(s);
  }
  /* restore saved typedef, enum and struct declarations */
  FOREACH(it, save.begin(), save.end()) {
    SgClassDeclaration *class_decl = isSgClassDeclaration(*it);
    if (class_decl) {
      si::fixStructDeclaration(class_decl, sc);
      continue;
    }
    si::appendStatement(isSgStatement(*it), sc);
  }
  /* replace original kernel functions */
  FOREACH(it, kernel_funcs.begin(), kernel_funcs.end()) {
    SgFunctionDeclaration *func = isSgFunctionDeclaration(*it);
    RunKernelAttribute *run_kernel_attr =
        rose_util::GetASTAttribute<RunKernelAttribute>(func);
    SgFunctionDeclaration *f =
        rose_util::CloneFunction(func, func->get_name(), sc);
    if (run_kernel_attr) {
      rose_util::AddASTAttribute(f, run_kernel_attr);
      f->get_declarationModifier().get_storageModifier().setStatic(); /* static */
      si::insertStatementAfterLastDeclaration(
          run_kernel_attr->stencil_map()->GetStencilTypeDefinition()->get_declaration(),
          sc);  /* stencil struct */
    } else if (rose_util::GetASTAttribute<RunKernelCallerAttribute>(func)) {
      f->get_declarationModifier().get_storageModifier().setExtern(); /* extern */
      f->set_linkage("C"); /* extern "C" */
    } else {
      f->get_declarationModifier().get_storageModifier().setStatic(); /* static */
    }
    si::insertStatementAfterLastDeclaration(f, sc);
  }
}
/** set dynamic link library file name in __dl_fname[],
 *  and attach '#include <dlfcn.h>'
 * @param[in] proj
 * @param[in] n ... number of dynamic link libraries
 * @param[in] dl_filename_suffix ... dynamic link library source suffix
 */
static void SetDynamicLinkLib(
    SgProject *proj, int n, string &dl_filename_suffix) {
  SgScopeStatement *sc = si::getFirstGlobalScope(proj);
  std::vector<SgExpression *> v;
  string s =
      dl_filename_suffix.substr(0, dl_filename_suffix.find_last_of('.')) +
      ".so";  /* '.??' --> '.so' */
  for (int i = 0; i < n; ++i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%05d.", i);
    string name = generate_output_filename(
        proj->get_fileList()[0]->get_sourceFileNameWithoutPath(),
        buf + s);
    v.push_back(sb::buildStringVal(name));
  }
  SgVariableDeclaration *vdecl = 
      sb::buildVariableDeclaration(
          "__dl_fname",
          sb::buildConstType(
              sb::buildArrayType(sb::buildPointerType(sb::buildCharType()))),
          sb::buildAssignInitializer(
              sb::buildAggregateInitializer(sb::buildExprListExp(v))),
          sc);
  si::setStatic(vdecl);
  //si::insertStatementBefore(si::findFirstDefiningFunctionDecl(sc), vdecl);
  si::prependStatement(vdecl, sc);
  si::attachArbitraryText(vdecl, "#include <dlfcn.h>");
}

} // namespace translator
} // namespace physis

// In Fortran, rmod files are read before the source file, so they
// need to be skipped.
static SgFile *GetMainSourceFile(SgProject *proj) {
  for (int i = 0; i < proj->numberOfFiles(); ++i) {
    SgFile *file = (*proj)[i];
    Sg_File_Info *finfo = file->get_file_info();
    if (physis::endswith(finfo->get_filenameString(), ".rmod"))
      continue;
    return file;
  }
  LOG_ERROR() << "No main source file found.\n";
  PSAbort(1);
  return NULL;
}

int main(int argc, char *argv[]) {
  pt::Translator *trans = NULL;
  string filename_target_suffix;
  string filename_suffix;
  string dl_filename_suffix;  /* dynamic link library source suffix */
  string extra_arg;
  pt::CommandLineOptions opts;
  std::vector<std::string> argvec;
  argvec.push_back(argv[0]);
  pt::parseOptions(argc, argv, opts, argvec);

  argvec.push_back("-DPHYSIS_USER");
  
  pt::Configuration config;
  if (opts.config_file_path.first) {
    if (opts.cuda_trans) {
      /* may auto tuning */
      config.LoadFile(opts.config_file_path.second, true);
    } else {
      /* no auto tuning */
      config.LoadFile(opts.config_file_path.second, false);
    }
    //LOG_DEBUG() << config << "\n";
    //LOG_DEBUG() << "auto_tuning: " << config.auto_tuning() << "\n";
    //LOG_DEBUG() << "pattern: " << config.npattern() << "\n";
    //LOG_DEBUG() << "dynamic: " << config.ndynamic() << "\n";
  }

  if (opts.ref_trans) {
    trans = new pt::ReferenceTranslator(config);
    filename_target_suffix = "ref";
    argvec.push_back("-DPHYSIS_REF");
  }

  if (opts.cuda_trans) {
    trans = new pt::CUDATranslator(config);
    filename_target_suffix = "cuda";
    argvec.push_back("-DPHYSIS_CUDA");
    /* set dynamic link library source suffix */
    dl_filename_suffix = "cuda_dl.cu";
  }
#ifdef CUDA_HM_TRANSLATOR_ENABLED
  if (opts.cuda_hm_trans) {
    trans = new pt::CUDAHMTranslator(config);
    filename_target_suffix = "cuda-hm";
    argvec.push_back("-DPHYSIS_CUDA_HM");
  }
#endif  

  if (opts.mpi_trans) {
    trans = new pt::MPITranslator(config);
    filename_target_suffix = "mpi";
    argvec.push_back("-DPHYSIS_MPI");
  }

  // if (opts.mpi2_trans) {
  //   trans = new pt::MPITranslator2(config);
  //   filename_suffix = "mpi2.c";
  //   argvec.push_back("-DPHYSIS_MPI");
  // }

#ifdef MPI_CUDA_TRANSLATOR_ENABLED
  if (opts.mpi_cuda_trans) {
    trans = new pt::MPICUDATranslator(config);
    filename_target_suffix = "mpi-cuda";
    argvec.push_back("-DPHYSIS_MPI_CUDA");
  }
#endif

#ifdef OPENCL_TRANSLATOR_ENABLED
  if (opts.opencl_trans) {
    trans = new pt::OpenCLTranslator(config);
    filename_target_suffix = "opencl";
    argvec.push_back("-DPHYSIS_OPENCL");
  }
#endif

#ifdef MPI_OPENCL_TRANSLATOR_ENABLED
  if (opts.mpi_opencl_trans) {
    trans = new pt::MPIOpenCLTranslator(config);
    filename_target_suffix = "mpi-opencl";
    argvec.push_back("-DPHYSIS_MPI_OPENCL");
  }
#endif

#ifdef MPI_OPENMP_TRANSLATOR_ENABLED
  if (opts.mpi_openmp_trans) {
    trans = new pt::MPIOpenMPTranslator(config);
    filename_suffix = "mpi-openmp.c";
    argvec.push_back("-DPHYSIS_MPI_OPENMP");
  }

  if (opts.mpi_openmp_numa_trans) {
    // Use the same MPIOpenMPTranslator
    trans = new pt::MPIOpenMPTranslator(config);
    filename_target_suffix = "mpi-openmp-numa";
    argvec.push_back("-DPHYSIS_MPI_OPENMP");
    argvec.push_back("-DPHYSIS_MPI_OPENMP_NUMA");
  }
#endif
  
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

  bool is_fortran = ru::IsFortran(proj);
  if (is_fortran) {
    LOG_DEBUG() << "Fortran input\n";
  }

  pt::TranslationContext tx(proj);
  pt::BuilderInterface *rt_builder = GetRTBuilder(proj, config, opts);
  pto::Optimizer *optimizer =
      GetOptimizer(&tx, proj, rt_builder, opts, &config);
  
  trans->SetUp(proj, &tx, rt_builder);

  LOG_INFO() << "Performing optimization Stage 1\n";
  // optimizer is null if not defined for this target
  if (optimizer) optimizer->Stage1();
  LOG_INFO() << "Optimization Stage 1 done\n";
  
  LOG_INFO() << "Translating the AST\n";
  trans->Translate();
  // TODO: optimization is disabled
  //trans->Optimize();
  LOG_INFO() << "Translation done\n";
  
  /* auto tuning & has dynamic link libraries */
  if (config.auto_tuning() && config.npattern() > 1) {
    LOG_INFO() << "Generating AT version.\n";
    
    pt::SetDynamicLinkLib(proj, config.npattern(), dl_filename_suffix);
    std::vector<SgFunctionDeclaration *> orig = pt::GetRunKernelFunc(proj);

    delete optimizer;

    pt::set_output_filename(GetMainSourceFile(proj), filename_suffix);

    int b = backend(proj);  /* without kernel function */
    LOG_INFO() << "Base code generation complete.\n";
    if (b) {
      LOG_ERROR() << "Backend failure.\n";
      trans->Finish();
      return b;
    }

    /* output dynamic link libraries */
    for (int i = 0; i < config.npattern(); ++i) {
      pt::ReplaceCloneRunKernelFunc(proj, orig);
      config.SetPat(i);

      optimizer = GetOptimizer(&tx, proj, rt_builder, opts, &config);
      LOG_INFO() << i << ":Performing optimization Stage 2\n";
      optimizer->Stage2();
      LOG_INFO() << i << ":Optimization Stage 2 done\n";
      delete optimizer;

      char buf[32];
#if 1 /* add optimize parameter as comment */
      string debug_comment = "\n";
      for (int ii = 0; !config.at_params_pattern[ii].empty(); ++ii) {
        snprintf(buf, sizeof(buf), "  %s = %d\n",
                 config.at_params_pattern[ii].c_str(),
                 config.LookupFlag(config.at_params_pattern[ii]));
        debug_comment += buf;
      }
      si::attachComment(si::getLastStatement(si::getFirstGlobalScope(proj)),
                        debug_comment, PreprocessingInfo::after);
#endif
      snprintf(buf, sizeof(buf), "%05d.", i);
      pt::set_output_filename(GetMainSourceFile(proj),
                              buf + dl_filename_suffix);
      b = backend(proj);  /* optimized kernel function */
      LOG_INFO() << i << ": Code generation complete.\n";
      if (b) {
        LOG_ERROR() << i << ": Backend failure.\n";
        trans->Finish();
        return b;
      }
    }
    LOG_DEBUG() << "AT code generation done.\n";
    trans->Finish();
    return b;
  }

  LOG_INFO() << "Performing optimization Stage 2\n";
  if (is_fortran) {
    LOG_WARNING() << "No optimization implemented for Fortran\n";
  } else if (optimizer) {
    optimizer->Stage2();
  } else {
    LOG_INFO() << "No optimizer defined\n";
  }
  LOG_INFO() << "Optimization Stage 2 done\n";
  
  trans->Finish();
  delete rt_builder;
  delete optimizer;

  filename_suffix = ru::GetInputFileSuffix(proj);
  if (opts.cuda_trans || opts.cuda_hm_trans || opts.mpi_cuda_trans) {
    filename_suffix = "cu";
    if (is_fortran) filename_suffix += "f";
  }

  pt::set_output_filename(
      GetMainSourceFile(proj),
      filename_target_suffix + "." + filename_suffix);

  int b = backend(proj);
  LOG_INFO() << "Code generation complete.\n";

  if (is_fortran) {
    pt::FixFortranOutput(
        GetMainSourceFile(proj)->get_unparse_output_filename());
  }
  
  return b;
}
