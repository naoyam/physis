! Copyright 2011-2013, RIKEN AICS.
! All rights reserved.
!
! This file is distributed under the BSD license. See LICENSE.txt for
! details.

#include "physis/fortran/physis.h"

module physis
  implicit none
  
  type ps_domain
     ps_index :: min, max
  end type ps_domain

  declare_grid_type(real, real)

  ! type :: ps_grid_real
  !    real :: pt
  ! end type ps_grid_real

  ! type :: ps_grid_double
  !    double precision :: pt
  ! end type ps_grid_double

  ! interface ps_grid_copyin
  !    subroutine ps_grid_copyin_real(grid, array)
  !      import :: ps_grid_real
  !      type(ps_grid_real), dimension(:,:,:), intent(out) :: grid
  !      real, dimension(:,:,:), intent(in) :: array
  !    end subroutine ps_grid_copyin_real
  ! end interface ps_grid_copyin

  ! interface ps_grid_copyout
  !    subroutine ps_grid_copyout_real(grid, array)
  !      import :: ps_grid_real
  !      type(ps_grid_real), dimension(:,:,:), intent(in) :: grid
  !      real, dimension(:,:,:), intent(out) :: array
  !    end subroutine ps_grid_copyout_real
  ! end interface ps_grid_copyout

  ! interface ps_grid_copyin
  !    subroutine ps_grid_copyin_double(grid, array)
  !      import :: ps_grid_double
  !      type(ps_grid_double), dimension(:,:,:), intent(out) :: grid
  !      double precision, dimension(:,:,:), intent(in) :: array
  !    end subroutine ps_grid_copyin_double
  ! end interface ps_grid_copyin

  ! interface ps_grid_copyout
  !    subroutine ps_grid_copyout_double(grid, array)
  !      import :: ps_grid_double
  !      type(ps_grid_double), dimension(:,:,:), intent(in) :: grid
  !      double precision, dimension(:,:,:), intent(out) :: array
  !    end subroutine ps_grid_copyout_double
  ! end interface ps_grid_copyout

  interface ps_stencil_map
  end interface ps_stencil_map

  
contains
  
  subroutine ps_init()
  end subroutine ps_init

  subroutine ps_finalize()
  end subroutine ps_finalize


end module physis
