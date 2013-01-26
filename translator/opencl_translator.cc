#include "translator/opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/SageBuilderEx.h"

namespace sbx = physis::translator::SageBuilderEx;
namespace si = SageInterface;
namespace sb = SageBuilder;

#define BLOCK_DIM_X_DEFAULT (64)
#define BLOCK_DIM_Y_DEFAULT (4)
#define BLOCK_DIM_Z_DEFAULT (1)

namespace physis {
namespace translator {

OpenCLTranslator::OpenCLTranslator(const Configuration &config):
    ReferenceTranslator(config),
    block_dim_x_(BLOCK_DIM_X_DEFAULT),
    block_dim_y_(BLOCK_DIM_Y_DEFAULT),
    block_dim_z_(BLOCK_DIM_Z_DEFAULT)
{
  // TODO: Need check & implementation

  target_specific_macro_ = "PHYSIS_OPENCL";  
  flag_pre_calc_grid_address_ = false;

#if 0
  const pu::LuaValue *lv
      = config.Lookup(Configuration::OPENCL_PRE_CALC_GRID_ADDRESS);
  if (lv) {
    PSAssert(lv->get(flag_pre_calc_grid_address_));
  }
#else
  const pu::LuaValue *lv;
#endif
  if (flag_pre_calc_grid_address_) {
    LOG_INFO() << "Optimization of address calculation enabled.\n";
  }

  // Redefine the block size if specified in the configuration file
  lv = config.Lookup(Configuration::OPENCL_BLOCK_SIZE);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    std::vector<double> v;
    PSAssert(tbl->get(v));
    block_dim_x_ = (int)v[0];
    block_dim_y_ = (int)v[1];
    block_dim_z_ = (int)v[2];
  }
  validate_ast_ = false;
} // OpenCLTranslator

void OpenCLTranslator::Finish()
{
  LOG_INFO() << "Adding #ifndef " << kernel_mode_macro() << "\n";
  std::string str_insert = "#ifndef ";
  str_insert += kernel_mode_macro();
  si::attachArbitraryText(
      src_->get_globalScope(),
      str_insert,
      PreprocessingInfo::before
                          );
  str_insert = "#endif /* #ifndef ";
  str_insert += kernel_mode_macro();
  str_insert += " */";
  si::attachArbitraryText(
      src_->get_globalScope(),
      str_insert,
      PreprocessingInfo::after
                          );

  add_opencl_extension_pragma();
  check_consistency();

} // run

void OpenCLTranslator::add_opencl_extension_pragma()
{
  // Add pragma for double case
  FOREACH(it, tx_->grid_new_map().begin(), tx_->grid_new_map().end()) {
    Grid *grid = it->second;
    GridType *gt = grid->getType();
    SgType *ty = gt->getElmType();
    if (! isSgTypeDouble(ty)) continue;

    LOG_INFO() << "Adding cl_khr_fp64 pragma for double support";

    std::string str_insert = "";
    str_insert = "#endif /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += " */\n";
    str_insert += "#ifdef ";
    str_insert += kernel_mode_macro();
    str_insert += "\n";
    str_insert += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
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
        PreprocessingInfo::before
                            );

    break;
  };

  return;

}

SgExpression *OpenCLTranslator::BuildBlockDimX()
{
  return sb::buildIntVal(block_dim_x_);
} // BuildBlockDimX

SgExpression *OpenCLTranslator::BuildBlockDimY()
{
  return sb::buildIntVal(block_dim_y_);  
} // BuildBlockDimY

SgExpression *OpenCLTranslator::BuildBlockDimZ()
{
  return sb::buildIntVal(block_dim_z_);
} // BuildBlockDimZ

SgType *OpenCLTranslator::BuildOnDeviceGridType(GridType *gt)
{
  PSAssert(gt);
  string gt_name;
  int nd = gt->getNumDim();
  string elm_name = gt->getElmType()->unparseToString();
  std::transform(elm_name.begin(), elm_name.begin() +1,
                 elm_name.begin(), toupper);
  string ondev_type_name = "__PSGrid" + toString(nd) + "D"
      + elm_name + "Dev";
  LOG_DEBUG() << "On device grid type name: "
              << ondev_type_name << "\n";
  SgType *t =
      si::lookupNamedTypeInParentScopes(ondev_type_name, global_scope_);
  PSAssert(t);

  return t;
} // BuildOnDeviceGridType

} // namespace translator
} // namespace physis

