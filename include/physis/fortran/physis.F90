! Copyright 2011-2013, RIKEN AICS.
! All rights reserved.
!
! This file is distributed under the BSD license. See LICENSE.txt for
! details.

#include "physis/fortran/physis.h"

module physis
  implicit none

  type PSDomain1D
     integer, dimension(1) :: local_min, local_max
  end type PSDomain1D
  type PSDomain2D
     integer, dimension(2) :: local_min, local_max
  end type PSDomain2D
  type PSDomain3D
     integer, dimension(3) :: local_min, local_max
  end type PSDomain3D

#if 0
  interface 
     type(PSDomain1D) function PSDomain1DNew(i1, i2)
       import PSDomain1D
       integer :: i1, i2
     end function PSDomain1DNew
     type(PSDomain2D) function PSDomain2DNew(i1, i2, j1, j2)
       import PSDomain2D
       integer :: i1, i2, j1, j2
     end function PSDomain2DNew
     type(PSDomain3D) function PSDomain3DNew(i1, i2, j1, j2, k1, k2)
       import PSDomain3D
       integer :: i1, i2, j1, j2, k1, k2
     end function PSDomain3DNew
  end interface
#endif
  

  type PSGrid3DReal
     real, dimension(:,:,:), allocatable :: pt
  end type PSGrid3DReal

  type PSGrid3DDouble
     double precision, dimension(:,:,:), allocatable :: pt
  end type PSGrid3DDouble

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
