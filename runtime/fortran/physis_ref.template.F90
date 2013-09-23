! Copyright 2011-2013, RIKEN AICS.
! All rights reserved.
!
! This file is distributed under the BSD license. See LICENSE.txt for
! details.

#use template_grid_type.f90
#use template_grid_func_ref.f90
#use template_domain.f90

module physis_ref
  implicit none

#DeclareDomainTypes()

#DeclareGridType1D(Real, real)
#DeclareGridType1D(Double, double precision)
#DeclareGridType2D(Real, real)
#DeclareGridType2D(Double, double precision)
#DeclareGridType3D(Real, real)
#DeclareGridType3D(Double, double precision)
  
  interface PSGridNew
     module procedure PSGridNew_3DReal
  end interface PSGridNew

  interface PSGridCopyin
     module procedure PSGridCopyin_3DReal
  end interface PSGridCopyin
  
  interface PSGridCopyout
     module procedure PSGridCopyout_3DReal
  end interface PSGridCopyout

  type PSStencil
     contains
       procedure :: run
  end type PSStencil

contains

  subroutine PSInit()
  end subroutine PSInit
  
  subroutine PSFinalize()
  end subroutine PSFinalize
  
  type(PSDomain1D) function PSDomain1DNew(i1, i2) result(dim)
    integer :: i1, i2
    dim%local_min(1) = i1
    dim%local_max(1) = i2
  end function PSDomain1DNew
  type(PSDomain2D) function PSDomain2DNew(i1, i2, j1, j2) result(dim)
    integer :: i1, i2, j1, j2
    dim%local_min = (/i1, j1/)
    dim%local_max = (/i2, j2/)
  end function PSDomain2DNew
  type(PSDomain3D) function PSDomain3DNew(i1, i2, j1, j2, k1, k2) result(dim)
    integer :: i1, i2, j1, j2, k1, k2
    dim%local_min = (/i1, j1, k1/)
    dim%local_max = (/i2, j2, k2/)
  end function PSDomain3DNew

#DeclareGridFuncs1D(Real, real)
#DeclareGridFuncs1D(Double, double precision)    
#DeclareGridFuncs2D(Real, real)
#DeclareGridFuncs2D(Double, double precision)    
#DeclareGridFuncs3D(Real, real)
#DeclareGridFuncs3D(Double, double precision)    

  ! dummy fucntion. not to be called. should be private
  subroutine run(s)
    class(PSStencil) :: s
  end subroutine run
  
  subroutine PSStencilRun(s, c)
    type(PSStencil), dimension(:) :: s
    integer c
    integer i, j
    do i = 1, C
       do j = 1, size(s, 1)
          call s(j)%run()
       end do
    end do
  end subroutine PSStencilRun

end module physis_ref
