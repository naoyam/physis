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
  
#DeclareGridFuncInterface(1, Real)
#DeclareGridFuncInterface(2, Real)
#DeclareGridFuncInterface(3, Real)  
#DeclareGridFuncInterface(1, Double)
#DeclareGridFuncInterface(2, Double)
#DeclareGridFuncInterface(3, Double)  

  type PSStencil
     contains
       procedure :: run
  end type PSStencil

  interface PSStencilRun
     module procedure PSStencilRun1, PSStencilRun2, PSStencilRun3,&
          PSStencilRun4, PSStencilRun5
  end interface PSStencilRun

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

  subroutine PSStencilRun1(s1, c)
    class(PSStencil), pointer :: s1
    integer c
    integer i
    do i = 1, C
       call s1%run()
    end do
  end subroutine PSStencilRun1

  subroutine PSStencilRun2(s1, s2, c)
    class(PSStencil), pointer :: s1, s2
    integer c
    integer i
    do i = 1, C
       call s1%run()
       call s2%run()       
    end do
  end subroutine PSStencilRun2

  subroutine PSStencilRun3(s1, s2, s3, c)
    class(PSStencil), pointer :: s1, s2, s3
    integer c
    integer i
    do i = 1, C
       call s1%run()
       call s2%run()
       call s3%run()              
    end do
  end subroutine PSStencilRun3

  subroutine PSStencilRun4(s1, s2, s3, s4, c)
    class(PSStencil), pointer :: s1, s2, s3, s4
    integer c
    integer i
    do i = 1, C
       call s1%run()
       call s2%run()
       call s3%run()
       call s4%run()                     
    end do
  end subroutine PSStencilRun4

  subroutine PSStencilRun5(s1, s2, s3, s4, s5, c)
    class(PSStencil), pointer :: s1, s2, s3, s4, s5
    integer c
    integer i
    do i = 1, C
       call s1%run()
       call s2%run()
       call s3%run()
       call s4%run()
       call s5%run()
    end do
  end subroutine PSStencilRun5
  
end module physis_ref
