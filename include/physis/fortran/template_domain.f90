#define DeclareDomainTypes()
  type PSDomain1D
     integer, dimension(1) :: local_min, local_max
  end type PSDomain1D
  type PSDomain2D
     integer, dimension(2) :: local_min, local_max
  end type PSDomain2D
  type PSDomain3D
     integer, dimension(3) :: local_min, local_max
  end type PSDomain3D
#end  
