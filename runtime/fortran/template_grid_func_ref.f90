#define DeclareGridFuncInterface(dim, name)
  interface PSGridNew
     module procedure PSGridNew_${dim}D${name}
  end interface PSGridNew
  interface PSGridCopyin
     module procedure PSGridCopyin_${dim}D${name}
  end interface PSGridCopyin
  interface PSGridCopyout
     module procedure PSGridCopyout_${dim}D${name}
  end interface PSGridCopyout
#end

#define DeclareGridFuncs1D(name, type)
  subroutine PSGridNew_1D${name}(g, x)
    type(PSGrid1D${name}) :: g
    integer, intent(in) :: x
    allocate(g%pt(x))
  end subroutine PSGridNew_1D${name}

  subroutine PSGridCopyin_1D${name}(g, a)
    type(PSGrid1D${name}) :: g
    ${type}, dimension(:), intent(in) :: a
    g%pt = a
  end subroutine PSGridCopyin_1D${name}

  subroutine PSGridCopyout_1D${name}(g, a)
    type(PSGrid1D${name}) :: g
    ${type}, dimension(:), intent(out) :: a
    a = g%pt
  end subroutine PSGridCopyout_1D${name}
#end

#define DeclareGridFuncs2D(name, type)  
  subroutine PSGridNew_2D${name}(g, x, y)
    type(PSGrid2D${name}) :: g
    integer, intent(in) :: x, y
    allocate(g%pt(x, y))
  end subroutine PSGridNew_2D${name}

  subroutine PSGridCopyin_2D${name}(g, a)
    type(PSGrid2D${name}) :: g
    ${type}, dimension(:,:), intent(in) :: a
    g%pt = a
  end subroutine PSGridCopyin_2D${name}

  subroutine PSGridCopyout_2D${name}(g, a)
    type(PSGrid2D${name}) :: g
    ${type}, dimension(:,:), intent(out) :: a
    a = g%pt
  end subroutine PSGridCopyout_2D${name}
#end

#define DeclareGridFuncs3D(name, type)    
  subroutine PSGridNew_3D${name}(g, x, y, z)
    type(PSGrid3D${name}) :: g
    integer, intent(in) :: x, y, z
    allocate(g%pt(x, y, z))
  end subroutine PSGridNew_3D${name}

  subroutine PSGridCopyin_3D${name}(g, a)
    type(PSGrid3D${name}) :: g
    ${type}, dimension(:,:,:), intent(in) :: a
    g%pt = a
  end subroutine PSGridCopyin_3D${name}

  subroutine PSGridCopyout_3D${name}(g, a)
    type(PSGrid3D${name}) :: g
    ${type}, dimension(:,:,:), intent(out) :: a
    a = g%pt
  end subroutine PSGridCopyout_3D${name}
#end  
  
