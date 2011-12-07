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

  const pu::LuaValue *lv
      = config.Lookup(Configuration::CUDA_PRE_CALC_GRID_ADDRESS);
  if (lv) {
    PSAssert(lv->get(flag_pre_calc_grid_address_));
  }
  if (flag_pre_calc_grid_address_) {
    LOG_INFO() << "Optimization of address calculation enabled.\n";
  }

  // Redefine the block size if specified in the configuration file
  lv = config.Lookup(Configuration::CUDA_BLOCK_SIZE);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    std::vector<double> v;
    PSAssert(tbl->get(v));
    block_dim_x_ = (int)v[0];
    block_dim_y_ = (int)v[1];
    block_dim_z_ = (int)v[2];
  }
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

} // run


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

