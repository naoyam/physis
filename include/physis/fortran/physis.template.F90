! Copyright 2011-2013, RIKEN AICS.
! All rights reserved.
!
! This file is distributed under the BSD license. See LICENSE.txt for
! details.

#use template_grid_type.f90
#use template_domain.f90

module physis
  implicit none
  
#DeclareDomainTypes()
  
  type PSStencil
  end type PSStencil

#DeclareGridType1D(Real, real)
#DeclareGridType1D(Double, double precision)
#DeclareGridType2D(Real, real)
#DeclareGridType2D(Double, double precision)
#DeclareGridType3D(Real, real)
#DeclareGridType3D(Double, double precision)

  external PSGridDim
  external PSGridNew
  external PSGridCopyin
  external PSGridCopyout
  external PSStencilMap
  external PSStencilRun
  
  interface
     subroutine PSInit()
     end subroutine PSInit
     subroutine PSFinalize()
     end subroutine PSFinalize
  end interface

contains

  type(PSDomain1D) function PSDomain1DNew(i1, i2)
    integer :: i1, i2
  end function PSDomain1DNew
  type(PSDomain2D) function PSDomain2DNew(i1, i2, j1, j2)
    integer :: i1, i2, j1, j2
  end function PSDomain2DNew
  type(PSDomain3D) function PSDomain3DNew(i1, i2, j1, j2, k1, k2)
    integer :: i1, i2, j1, j2, k1, k2
  end function PSDomain3DNew
  
end module physis
