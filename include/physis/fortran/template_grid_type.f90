  
#define DeclareGridType1D(name, type)
  type PSGrid1D${name}
     ${type}, dimension(:), allocatable :: pt
  end type PSGrid1D${name}
#end

#define DeclareGridType2D(name, type)
  type PSGrid2D${name}
     ${type}, dimension(:,:), allocatable :: pt
  end type PSGrid2D${name}
#end
  
#define DeclareGridType3D(name, type)
  type PSGrid3D${name}
     ${type}, dimension(:,:,:), allocatable :: pt
  end type PSGrid3D${name}
#end
