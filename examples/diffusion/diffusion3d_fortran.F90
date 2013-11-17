#include "physis/fortran/physis.h"

#ifndef NN
#define NN (32)
#endif
#ifndef COUNT
#define COUNT (100)
#endif

module diffusion3d_kernel
  use physis
  implicit none
  
contains
  subroutine diffusion(i, j, k, p0, p1, nx, ny, nz, &
       ce, cw, cn, cs, ct, cb, cc)
    implicit none
    !real, dimension(:, :, :), intent(inout) :: p0, p1
    type(PSGrid3DReal), intent(inout) :: p0, p1    
    integer, intent(in) :: i, j, k
    integer, intent(in) :: nx, ny, nz
    real, intent(in) :: ce, cw, cn, cs, ct, cb, cc
    real :: c, w, e, s, n, b, t

    c = p0%pt(i, j, k)
    if (i == 1) then
       w = c
    else
       w = p0%pt(i-1, j, k)
    end if
    if (i == nx) then
       e = c
    else
       e = p0%pt(i+1, j, k)
    end if
    if (j == 1) then
       s = c
    else
       s = p0%pt(i, j-1, k)
    end if
    if (j == ny) then
       n = c
    else
       n = p0%pt(i, j+1, k)
    end if
    if (k == 1) then
       b = c
    else
       b = p0%pt(i, j, k-1)
    end if
    if (k == nz) then
       t = c
    else
       t = p0%pt(i, j, k+1)
    end if
    p1%pt(i, j, k) = cc * c + cw * w + ce * e + &
         cs * s + cn * n + cb * b + ct * t
  end subroutine diffusion
  
end module diffusion3d_kernel

program diffusion3d
  use physis
  use diffusion3d_kernel

  implicit none
  
  integer, parameter :: nx = NN, ny = NN, nz = NN
  real, parameter :: M_PI = 3.1415926535897932384626
  real, dimension(nx, ny, nz) :: p0
  real :: ce, cw, cn, cs, ct, cb, cc
  real :: dt, kappa, dx, dy, dz, kx, ky, kz
  integer :: i
  integer :: clock_start, clock_end, clock_rate
  type(PSGrid3DReal) :: p0d, p1d
  !type(ps_grid_real) :: p0d, p1d
  type(PSDomain3D) :: dom
  type(PSStencil), pointer :: s1, s2
  
  kappa = 0.1
  dx = 1.0 / nx
  dy = 1.0 / ny
  dz = 1.0 / nz
  kx = 2.0 * M_PI
  ky = 2.0 * M_PI
  kz = 2.0 * M_PI 
  dt = 0.1 * dx * dx / kappa
  ce = kappa*dt/(dx*dx)
  cw = kappa*dt/(dx*dx)
  cn = kappa*dt/(dy*dy);
  cs = kappa*dt/(dy*dy);
  ct = kappa*dt/(dz*dz);
  cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb)

  call initialize(p0, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, 0.0)
  call PSInit()

  call PSGridNew(p0d, nx, ny, nz)
  call PSGridNew(p1d, nx, ny, nz)  

  dom = PSDomain3Dnew(1, nx, 1, ny, 1, nz)

  call PSGridCopyin(p0d, p0)

  if (mod(COUNT, 2) /= 0) then
     write (*,*) "Iteration count:", COUNT
     write (*,*) "Iteration count must be an even number."
     stop
  end if

  call PSStencilMap(s1, diffusion, dom, &
       p0d, p1d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc)
  call PSStencilMap(s2, diffusion, dom, &
       p1d, p0d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc)
  
  call system_clock(clock_start, clock_rate)
  call PSStencilRun(s1, s2, COUNT / 2)
  call system_clock(clock_end, clock_rate)

  call PSGridCopyout(p0d, p0)
  
  !call dump(p0, nx, ny, nz)
  write (*,"(a, f10.3)") "Elapsed time:", (clock_end - clock_start) / real(clock_rate)
  write (*,"(a, E15.7)") "Accurcy: ", get_accuracy(p0, nx, ny, nz, kx, ky, kz, &
       dx, dy, dz, kappa, dt * COUNT)

  stop
  
contains

  ! initialize  
  subroutine initialize(p, nx, ny, nz, &
       kx, ky, kz, dx, dy, dz, kappa, time)
    implicit none
    integer :: nx, ny, nz
    real, dimension(nx, ny, nz) :: p
    real :: time, kappa, dx, dy, dz, kx, ky, kz
    real :: x, y, z
    real :: ax, ay, az
    real :: f0
    integer :: i, j, k

    ax = exp(-kappa*time*(kx*kx))
    ay = exp(-kappa*time*(ky*ky))
    az = exp(-kappa*time*(kz*kz))
      
    do k = 1, nz
       do j = 1, ny
          do i = 1, nx
             x = dx * (i - 0.5)
             y = dy * (j - 0.5)
             z = dz * (k - 0.5)
             f0 = 0.125 * (1.0-ax*cos(kx*x)) &
                  * (1.0-ay*cos(ky*y)) &
                  * (1.0-az*cos(kz*z))
             p(i, j, k) = f0
          end do
       end do
    end do
    
  end subroutine initialize

  subroutine dump(p, nx, ny, nz)
    implicit none
    integer :: nx, ny, nz
    real, dimension(nx, ny, nz) :: p
    integer :: i, j, k

    do k = 1, nz
       do j = 1, ny
          do i = 1, nx
             write (*,'(e15.7)') p(i, j, k)
          end do
       end do
    end do
    
  end subroutine dump

  real function get_accuracy(p, nx, ny, nz, &
       kx, ky, kz, dx, dy, dz, kappa, dt)
    implicit none
    integer, intent(in) :: nx, ny, nz
    real, intent(in) :: kx, ky, kz, dx, dy, dz, kappa, dt    
    real, dimension(nx, ny, nz), intent(in) :: p
    real, dimension(nx, ny, nz) :: ref
    real :: err, diff
    integer :: i, j, k
    
    call initialize(ref, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, dt)

    err = 0.0;
    do k = 1, nz
       do j = 1, ny
          do i = 1, nx
             diff = ref(i, j, k) - p(i, j, k)
             err = err + diff * diff;
          end do
       end do
    end do
    get_accuracy = sqrt(err/(nx * ny * nz))
    
  end function get_accuracy
  
end program diffusion3d
