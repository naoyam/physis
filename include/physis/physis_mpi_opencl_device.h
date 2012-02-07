#ifndef PHYSIS_PHYSIS_MPI_OPENCL_DEVICE_H_
#define PHYSIS_PHYSIS_MPI_OPENCL_DEVICE_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PS_SIZE_T
#ifdef PHYSIS_MPI_OPENCL_KERNEL_MODE
/* No size_t here */
/* typedef here doesn't seem to work */
#define PS_SIZE_T unsigned long int
#define PS_GLFT_P __global float *
#else
typedef size_t PS_SIZE_T
typedef float * PS_GLFT_P
#endif /* PHYSIS_MPI_OPENCL_KERNEL_MODE */
#endif /* PS_SIZE_T */

#if defined(PHYSIS_MPI_OPENCL_KERNEL_MODE)
#endif /* #if defined(PHYSIS_MPI_OPENCL_KERNEL_MODE) */ 

typedef struct {
    PS_GLFT_P p0;
    int dim[3];
    int local_size[3];
    int local_offset[3]; 
    int pitch;
    PS_GLFT_P halo[3][2];  
    int halo_width[3][2];    
    int diag;    
} __PSGrid3DFloatDev_CLKernel;

#define __PS_CL_ARG_EXPAND_ELEMENT_G(g) \
  __PS_ ## g ## _p0, \
  __PS_ ## g ## _dim_0, __PS_ ## g ## _dim_1, __PS_ ## g ## _dim_2, \
  __PS_ ## g ## _local_size_0, __PS_ ## g ## _local_size_1, \
      __PS_ ## g ## _local_size_2, \
  __PS_ ## g ## _local_offset_0, __PS_ ## g ## _local_offset_1, \
      __PS_ ## g ## _local_offset_2, \
  __PS_ ## g ## _pitch, \
  __PS_ ## g ## _halo_00, __PS_ ## g ## _halo_00_nonnull_p, \
  __PS_ ## g ## _halo_01, __PS_ ## g ## _halo_01_nonnull_p, \
  __PS_ ## g ## _halo_10, __PS_ ## g ## _halo_10_nonnull_p, \
  __PS_ ## g ## _halo_11, __PS_ ## g ## _halo_11_nonnull_p, \
  __PS_ ## g ## _halo_20, __PS_ ## g ## _halo_20_nonnull_p, \
  __PS_ ## g ## _halo_21, __PS_ ## g ## _halo_21_nonnull_p, \
  __PS_ ## g ## _halo_width_00, __PS_ ## g ## _halo_width_01, \
  __PS_ ## g ## _halo_width_10, __PS_ ## g ## _halo_width_11, \
  __PS_ ## g ## _halo_width_20, __PS_ ## g ## _halo_width_21, \
  __PS_ ## g ## _diag

#define __PS_CL_ARG_EXPAND_ELEMENT_G_WITH_TYPE(g) \
  PS_GLFT_P __PS_ ## g ## _p0, \
  long __PS_ ## g ## _dim_0, long __PS_ ## g ## _dim_1, \
      long __PS_ ## g ## _dim_2, \
  long __PS_ ## g ## _local_size_0, long __PS_ ## g ## _local_size_1, \
      long __PS_ ## g ## _local_size_2, \
  long __PS_ ## g ## _local_offset_0, long __PS_ ## g ## _local_offset_1, \
      long __PS_ ## g ## _local_offset_2, \
  long __PS_ ## g ## _pitch, \
  PS_GLFT_P __PS_ ## g ## _halo_00, long __PS_ ## g ## _halo_00_nonnull_p, \
  PS_GLFT_P __PS_ ## g ## _halo_01, long __PS_ ## g ## _halo_01_nonnull_p, \
  PS_GLFT_P __PS_ ## g ## _halo_10, long __PS_ ## g ## _halo_10_nonnull_p, \
  PS_GLFT_P __PS_ ## g ## _halo_11, long __PS_ ## g ## _halo_11_nonnull_p, \
  PS_GLFT_P __PS_ ## g ## _halo_20, long __PS_ ## g ## _halo_20_nonnull_p, \
  PS_GLFT_P __PS_ ## g ## _halo_21, long __PS_ ## g ## _halo_21_nonnull_p, \
  long __PS_ ## g ## _halo_width_00, long __PS_ ## g ## _halo_width_01, \
  long __PS_ ## g ## _halo_width_10, long __PS_ ## g ## _halo_width_11, \
  long __PS_ ## g ## _halo_width_20, long __PS_ ## g ## _halo_width_21, \
  long __PS_ ## g ## _diag

#define __PS_CL_ARG_EXPAND_ELEMENT_DOM(dom) \
  __PS_ ## dom ## _xmin, __PS_ ## dom ## _xmax, \
  __PS_ ## dom ## _ymin, __PS_ ## dom ## _ymax, \
  __PS_ ## dom ## _zmin, __PS_ ## dom ## _zmax

#define __PS_CL_ARG_EXPAND_ELEMENT_DOM_WITH_TYPE(dom) \
  long __PS_ ## dom ## _xmin, long __PS_ ## dom ## _xmax, \
  long __PS_ ## dom ## _ymin, long __PS_ ## dom ## _ymax, \
  long __PS_ ## dom ## _zmin, long __PS_ ## dom ## _zmax


#define __PS_ST_K __PSGrid3DFloatDev_CLKernel

#define __PS_INIT_XYZ(var, elm) \
    var -> elm [0] = __PS_ ## var ## _ ## elm ## _0 ;\
    var -> elm [1] = __PS_ ## var ## _ ## elm ## _1 ;\
    var -> elm [2] = __PS_ ## var ## _ ## elm ## _2

#define __PS_INIT_XYZ_FB(var, elm) \
    var -> elm [0][0] = __PS_ ## var ## _ ## elm ## _00 ;\
    var -> elm [0][1] = __PS_ ## var ## _ ## elm ## _01 ;\
    var -> elm [1][0] = __PS_ ## var ## _ ## elm ## _10 ;\
    var -> elm [1][1] = __PS_ ## var ## _ ## elm ## _11 ;\
    var -> elm [2][0] = __PS_ ## var ## _ ## elm ## _20 ;\
    var -> elm [2][1] = __PS_ ## var ## _ ## elm ## _21

#if 0
#define __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, XX, YY) \
  do {\
    if ( __PS_ ## var ## _ ## elm ## _ ## XX ## YY ## _nonnull_p ) \
      var -> elm [XX][YY] = __PS_ ## var ## _ ## elm ## _ ## XX ## YY ;\
  } while (0)
#else
#define __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, XX, YY) \
  do {\
    if ( __PS_ ## var ## _ ## elm ## _ ## XX ## YY ## _nonnull_p ) \
      var -> elm [XX][YY] = __PS_ ## var ## _ ## elm ## _ ## XX ## YY ;\
    else \
      var -> elm [XX][YY] = 0 ; \
  } while (0)
#endif

#define __PS_INIT_XYZ_FB_WITH_FLAG(var, elm) \
  __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, 0, 0) ; \
  __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, 0, 1) ; \
  __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, 1, 0) ; \
  __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, 1, 1) ; \
  __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, 2, 0) ; \
  __PS_INIT_XYZ_FB_WITH_FLAG_BASE(var, elm, 2, 1)
    

  void
  __PS_CL_construct_PSGrid_from_arg(
    __PS_ST_K *g,
    __PS_CL_ARG_EXPAND_ELEMENT_G_WITH_TYPE(g)
  ) {
      g->p0 = __PS_g_p0;
      __PS_INIT_XYZ(g, dim);
      __PS_INIT_XYZ(g, local_size);
      __PS_INIT_XYZ(g, local_offset);
      g->pitch = __PS_g_pitch;
      /*__PS_INIT_XYZ_FB(g, halo);*/
      __PS_INIT_XYZ_FB_WITH_FLAG(g, halo);
      __PS_INIT_XYZ_FB(g, halo_width);
      g->diag = __PS_g_diag;
      
  }

#undef __PS_INIT_XYZ
#undef __PS_INIT_XYZ_PB

typedef struct {
    PS_SIZE_T local_min[3];
    PS_SIZE_T local_max[3];
} __PSDomain_CLKernel;

  void
  __PS_CL_construct_PSDomain_from_arg(
    __PSDomain_CLKernel *dom,
    __PS_CL_ARG_EXPAND_ELEMENT_DOM_WITH_TYPE(dom)
  ) {
    dom->local_min[0] = __PS_dom_xmin;
    dom->local_min[1] = __PS_dom_ymin;
    dom->local_min[2] = __PS_dom_zmin;
    dom->local_max[0] = __PS_dom_xmax;
    dom->local_max[1] = __PS_dom_ymax;
    dom->local_max[2] = __PS_dom_zmax;
  }

  PS_SIZE_T __PSGridCalcOffset3D(
      int x, int y, int z,
      int pitch, int dimy
  ) {
    return x + y * pitch + z * pitch * dimy;
  }

  PS_GLFT_P __PSGridGetAddrNoHaloFloat3D(
    __PS_ST_K *g,
    int x, int y, int z
  ) {
    x -= g->local_offset[0];
    y -= g->local_offset[1];
    z -= g->local_offset[2];
    return g->p0 + __PSGridCalcOffset3D(
        x, y, z, g->pitch, g->local_size[1]);    
  }

  PS_GLFT_P __PSGridGetAddrNoHaloFloat3DLocal(
      __PS_ST_K *g,
      int x, int y, int z
  ) {
    return g->p0 + __PSGridCalcOffset3D(
        x, y, z, g->pitch, g->local_size[1]);    
  }
  
  PS_GLFT_P __PSGridEmitAddrFloat3D(
      __PS_ST_K *g,
      int x, int y, int z
    ) {
    x -= g->local_offset[0];
    y -= g->local_offset[1];
    z -= g->local_offset[2];
    return g->p0 + __PSGridCalcOffset3D(
          x, y, z, g->pitch,
          g->local_size[1]);    
  }

  // z
  PS_GLFT_P __PSGridGetAddrFloat3D_2_fw(
      __PS_ST_K *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[2] < g->local_size[2]) {
      return __PSGridGetAddrNoHaloFloat3DLocal(
          g, indices[0], indices[1], indices[2]);
    } else {
      indices[2] -= g->local_size[2];
      PS_SIZE_T offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->local_size[0], g->local_size[1]);
      return g->halo[2][1] + offset;
    }
  }


  PS_GLFT_P __PSGridGetAddrFloat3D_2_bw(
      __PS_ST_K *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[2] >= 0) {
      return __PSGridGetAddrNoHaloFloat3DLocal(
          g, indices[0], indices[1], indices[2]);
    } else {      
      indices[2] += g->halo_width[2][0];
      PS_SIZE_T offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
         g->local_size[0], g->local_size[1]);
      return g->halo[2][0] + offset;
    }
  }

  // y
  PS_GLFT_P __PSGridGetAddrFloat3D_1_fw(
      __PS_ST_K *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[1] < g->local_size[1]) {
      if (indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);        
      }
    } else {
      if (g->diag) indices[2] += g->halo_width[2][0];        
      indices[1] -= g->local_size[1];
      PS_SIZE_T offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->local_size[0], g->halo_width[1][1]);
      return g->halo[1][1] + offset;
    }
  }

  PS_GLFT_P __PSGridGetAddrFloat3D_1_bw(
      __PS_ST_K *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[1] >= 0) {
      if (indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);        
      }
    } else {
      if (g->diag) indices[2] += g->halo_width[2][0];        
      indices[1] += g->halo_width[1][0];
      PS_SIZE_T offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->local_size[0], g->halo_width[1][0]);          
      return g->halo[1][0] + offset;
    }
  }

  // x
  PS_GLFT_P __PSGridGetAddrFloat3D_0_fw(
      __PS_ST_K *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[0] < g->local_size[0]) {
      // not in the halo region of this dimension
      if (indices[1] < g->local_size[1] &&
          indices[1] >= 0 &&
          indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        // must be inside region
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[1] >= g->local_size[1]) {
        return __PSGridGetAddrFloat3D_1_fw(g, x, y, z);
      } else if (indices[1] < 0) {
        return __PSGridGetAddrFloat3D_1_bw(g, x, y, z);        
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);
      }
    } else {
      PS_SIZE_T halo_size1 = g->local_size[1];
      if (g->diag) {
        indices[1] += g->halo_width[1][0];
        indices[2] += g->halo_width[2][0];        
        halo_size1 += g->halo_width[1][0] +
            g->halo_width[1][1];
      }
      indices[0] -= g->local_size[0];
      PS_SIZE_T offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->halo_width[0][1], halo_size1);
      return g->halo[0][1] + offset;
    }
   }
  
  PS_GLFT_P __PSGridGetAddrFloat3D_0_bw(
      __PS_ST_K *g, int x, int y, int z) {
    int indices[] = {x - g->local_offset[0], y - g->local_offset[1],
		     z - g->local_offset[2]};
    if (indices[0] >= 0) { // not in the halo region of this dimension
      if (indices[1] < g->local_size[1] &&
          indices[1] >= 0 &&
          indices[2] < g->local_size[2] &&
          indices[2] >= 0) {
        // must be inside region
        return __PSGridGetAddrNoHaloFloat3DLocal(
            g, indices[0], indices[1], indices[2]);
      } else if (indices[1] >= g->local_size[1]) {
        return __PSGridGetAddrFloat3D_1_fw(g, x, y, z);
      } else if (indices[1] < 0) {
        return __PSGridGetAddrFloat3D_1_bw(g, x, y, z);        
      } else if (indices[2] >= g->local_size[2]) {
        return __PSGridGetAddrFloat3D_2_fw(g, x, y, z);
      } else {
        return __PSGridGetAddrFloat3D_2_bw(g, x, y, z);
      }
    } else {
      PS_SIZE_T halo_size1 = g->local_size[1];      
      if (g->diag) {
        indices[1] += g->halo_width[1][0];
        indices[2] += g->halo_width[2][0];        
        halo_size1 += g->halo_width[1][0] +
            g->halo_width[1][1];
      }
      indices[0] += g->halo_width[0][0];
      PS_SIZE_T offset = __PSGridCalcOffset3D(
          indices[0], indices[1], indices[2],
          g->halo_width[0][0], halo_size1);
      return g->halo[0][0] + offset;
    }
  }

  PS_GLFT_P __PSGridGetAddrFloat3D(__PS_ST_K *g,
                                 int x, int y, int z) {
    int indices[] = {x - g->local_offset[0],
		     y - g->local_offset[1],
		     z - g->local_offset[2]};
    PS_SIZE_T halo_size[3] = {g->local_size[0], g->local_size[1],
                           g->local_size[2]};          
    for (int i = 0; i < 3; ++i) {
      if (indices[i] < 0 || indices[i] >= g->local_size[i]) {
        PS_GLFT_P buf;
        if (g->diag) {
          for (int j = i+1; j < 3; ++j) {
            indices[j] += g->halo_width[j][0];
            halo_size[j] += g->halo_width[j][0] +
                            g->halo_width[j][1];
          }
        }
        PS_SIZE_T offset;
        if (indices[i] < 0) {
          indices[i] += g->halo_width[i][0];
          halo_size[i] = g->halo_width[i][0];
          buf = g->halo[i][0];
        } else {
          indices[i] -= g->local_size[i];
          halo_size[i] = g->halo_width[i][1];
          buf = g->halo[i][1];
        }
        offset = __PSGridCalcOffset3D(
				      indices[0], indices[1], indices[2],
				      halo_size[0], halo_size[1]);
        return buf + offset;
      }
    }
    return __PSGridGetAddrNoHaloFloat3D(g, x, y, z);
  }

#undef __PS_ST_K

#ifdef __cplusplus
}
#endif

#endif /* PHYSIS_PHYSIS_MPI_OPENCL_DEVICE_H_ */
