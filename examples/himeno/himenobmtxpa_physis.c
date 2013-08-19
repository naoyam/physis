/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 [mimax][mjmax][mkmax]:
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can 
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include "physis/physis.h"

//#define MR(mt,n,r,c,d)  mt->m[(n) * mt->mrows * mt->mcols * mt->mdeps + (r) * mt->mcols* mt->mdeps + (c) * mt->mdeps + (d)]
/*
  struct Mat {
  float* m;
  int mnums;
  int mrows;
  int mcols;
  int mdeps;
  };
*/
/* prototypes */
//typedef struct Mat Matrix;

void clearMat(PSGrid3DFloat Mat);
void set_param(int i[],char *size);
void mat_set(PSGrid3DFloat Mat, float z, float *buf);
void mat_set_init(PSGrid3DFloat Mat, float *buf);
float jacobi(int nn, PSGrid3DFloat a0, PSGrid3DFloat a1,
             PSGrid3DFloat a2, PSGrid3DFloat a3, PSGrid3DFloat b0,
             PSGrid3DFloat b1, PSGrid3DFloat b2,
             PSGrid3DFloat c0, PSGrid3DFloat c1, PSGrid3DFloat c2,
             PSGrid3DFloat p0, PSGrid3DFloat p1,
             PSGrid3DFloat bnd, PSGrid3DFloat wrk1);
double fflop(int,int,int);
double mflops(int,double,double);
double second();
double GetThroughput(int nn, int mx, int my, int mz, double time);

float   omega=0.8;
//Matrix  a,b,c,p,bnd,wrk1,wrk2;
int imax,jmax,kmax, mimax,mjmax,mkmax;
int
main(int argc, char *argv[])
{
  int    nn;
  int    msize[3];
  float  gosa,target;
  double  cpu0,cpu1,cpu,flop;
  char   *size;

#if 0
  size = "XS";
  set_param(msize,size);
  mimax= msize[0];
  mjmax= msize[1];
  mkmax= msize[2];
  imax= mimax-1;
  jmax= mjmax-1;
  kmax= mkmax-1;

  target = 60.0;
  
  PSInit(&argc, &argv, 3, mimax, mjmax, mkmax);
#else

  if (argc < 2) {
    printf("Error: problem size not specified\n");
    exit(1);
  }
  size = argv[1];
  set_param(msize,size);
  mimax= msize[0];
  mjmax= msize[1];
  mkmax= msize[2];
  imax= mimax-1;
  jmax= mjmax-1;
  kmax= mkmax-1;

  target = 60.0;
  PSInit(&argc, &argv, 3, mimax, mjmax, mkmax);  
#endif  

  printf("mimax = %d mjmax = %d mkmax = %d\n",mimax,mjmax,mkmax);
  printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);
  /*
   *    Initializing matrixes
   */
  PSGrid3DFloat p0 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat p1 = PSGrid3DFloatNew(mimax, mjmax, mkmax);  
  PSGrid3DFloat bnd = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat wrk1 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat a0 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat a1 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat a2 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat a3 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat b0 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat b1 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat b2 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat c0 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat c1 = PSGrid3DFloatNew(mimax, mjmax, mkmax);
  PSGrid3DFloat c2 = PSGrid3DFloatNew(mimax, mjmax, mkmax);

  float *host_buf = (float*)malloc(mimax *  mjmax  *mkmax
                                   * sizeof(float));
  
  mat_set_init(p0, host_buf);
  mat_set_init(p1, host_buf);  
  mat_set(bnd, 1.0, host_buf);
  //mat_set(wrk1, 0.0, host_buf);    
  mat_set(a0, 1.0, host_buf);
  mat_set(a1, 1.0, host_buf);
  mat_set(a2, 1.0, host_buf);
  mat_set(a3, 1.0/6.0, host_buf);
  mat_set(b0, 0.0, host_buf);
  mat_set(b1, 0.0, host_buf);
  mat_set(b2, 0.0, host_buf);
  mat_set(c0, 1.0, host_buf);
  mat_set(c1, 1.0, host_buf);
  mat_set(c2, 1.0, host_buf);
  free(host_buf);

  /*
   *    Start measuring
   */
  nn= 4;
  printf(" Start rehearsal measurement process.\n");
  printf(" Measure the performance in %d times.\n\n",nn);

  cpu0 = second();
  gosa = jacobi(nn, a0, a1, a2, a3, b0, b1, b2, c0, c1, c2,
                p0, p1, bnd, wrk1);
  cpu1 = second();
  cpu = cpu1 - cpu0;
  flop = fflop(imax,jmax,kmax);

  printf(" MFLOPS: %f time(s): %f %e\n\n",
         mflops(nn,cpu,flop),cpu,gosa);

  nn= (int)(target/(cpu/3.0));
  //nn=100;
  
  // make it even
  if (nn%2) ++nn;

#ifdef ENABLE_DUMP
  {
    // Dump p for correctness checking
    float *buf = (float*)malloc(sizeof(float) * mimax *mjmax * mkmax);
    PSGridCopyout(p0, buf);
    FILE* hout = fopen("himeno.physis.dat", "w");
    int i, j, k;
    int idx = 0;
    for (k = 0; k < mkmax; ++k) {
      for (j = 0; j < mjmax; ++j) {
        for (i = 0; i < mimax; ++i) {
          fprintf(hout, "%f\n", buf[idx]);
          ++idx;
        }
      }
    }
    fclose(hout);
  }
#endif  

  printf(" Now, start the actual measurement process.\n");
  printf(" The loop will be excuted in %d times\n",nn);
  printf(" This will take about one minute.\n");
  printf(" Wait for a while\n\n");

  cpu0 = second();
  gosa = jacobi(nn, a0, a1, a2, a3, b0, b1, b2, c0, c1, c2,
                p0, p1, bnd, wrk1);
  cpu1 = second();
  cpu = cpu1 - cpu0;


  printf(" Loop executed for %d times\n",nn);
  printf(" Gosa : %e \n",gosa);
  printf(" MFLOPS measured : %f\tcpu : %f\n",mflops(nn,cpu,flop),cpu);
  printf(" Throughput   : %.3f (GB/s)\n",
         GetThroughput(nn, mimax, mjmax, mkmax, cpu));
  
  printf(" Score based on Pentium III 600MHz using Fortran 77: %f\n",
         mflops(nn,cpu,flop)/82);

  clearMat(bnd);
  clearMat(wrk1);

  clearMat(a0);
  clearMat(a1);
  clearMat(a2);
  clearMat(a3);
  clearMat(b0);
  clearMat(b1);
  clearMat(b2);
  clearMat(c0);
  clearMat(c1);
  clearMat(c2);

  PSFinalize();
  return (0);
}

double
fflop(int mx,int my, int mz)
{
#ifdef COMPUTE_GOSA_EVERY_ITERATION
  return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*34.0);
#else
  return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*32.0);
#endif
}

double
mflops(int nn,double cpu,double flop)
{
  return(flop/cpu*1.e-6*(double)nn);
}

void
set_param(int is[],char *size)
{
  if(!strcmp(size,"XS") || !strcmp(size,"xs")){
    is[0]= 64;
    is[1]= 32;
    is[2]= 32;
    return;
  }
  if(!strcmp(size,"S") || !strcmp(size,"s")){
    is[0]= 128;
    is[1]= 64;
    is[2]= 64;
    return;
  }
  if(!strcmp(size,"M") || !strcmp(size,"m")){
    is[0]= 256;
    is[1]= 128;
    is[2]= 128;
    return;
  }
  if(!strcmp(size,"L") || !strcmp(size,"l")){
    is[0]= 512;
    is[1]= 256;
    is[2]= 256;
    return;
  }
  if(!strcmp(size,"XL") || !strcmp(size,"xl")){
    is[0]= 1024;
    is[1]= 512;
    is[2]= 512;
    return;
  } else {
    //printf("Invalid input character !!\n");
    //exit(6);
  }
}

void
clearMat(PSGrid3DFloat Mat)
{
  PSGridFree(Mat);
  return;
}

void
mat_set(PSGrid3DFloat mat, float val, float *buf)
{
  int i,j,k;

  size_t x = 0;
  for(i=0; i< PSGridDim(mat, 0); i++)
    for(j=0; j< PSGridDim(mat, 1); j++)
      for(k=0; k< PSGridDim(mat, 2); k++) {
        buf[x] = val;
        ++x;
      }
  
  PSGridCopyin(mat, buf);
}

void
mat_set_init(PSGrid3DFloat Mat, float *buf)
{
  int  i,j,k;

  int d0 = PSGridDim(Mat, 2);
  size_t x = 0;
  for(k=0; k< PSGridDim(Mat, 2); k++) 
    for(j=0; j< PSGridDim(Mat, 1); j++)        
      for(i=0; i< PSGridDim(Mat, 0); i++) {
        float v = (float)(k*k) / ((d0 - 1) * (d0 - 1));
        buf[x] = v;
        ++x;
      }

  PSGridCopyin(Mat, buf);
}

void jacobi_kernel(int i, int j, int k,
                   PSGrid3DFloat p0, PSGrid3DFloat p1,
                   PSGrid3DFloat a0, PSGrid3DFloat a1, PSGrid3DFloat a2,
                   PSGrid3DFloat a3, PSGrid3DFloat b0, PSGrid3DFloat b1,
                   PSGrid3DFloat b2, PSGrid3DFloat c0, PSGrid3DFloat c1,
                   PSGrid3DFloat c2, PSGrid3DFloat bnd, PSGrid3DFloat wrk1,
                   float omega)
{
  float s0, ss;
  s0= PSGridGet(a0, i, j, k) * PSGridGet(p0, i, j, k+1)
      + PSGridGet(a1, i, j, k) * PSGridGet(p0, i, j+1, k)
      + PSGridGet(a2, i, j, k) * PSGridGet(p0, i+1, j, k)
      + PSGridGet(b0, i, j, k)
      *( PSGridGet(p0, i, j+1, k+1) - PSGridGet(p0, i, j-1, k+1)
         - PSGridGet(p0, i, j+1, k-1) + PSGridGet(p0, i, j-1, k-1) )
      + PSGridGet(b1, i, j, k)
      *( PSGridGet(p0, i+1, j+1, k) - PSGridGet(p0, i+1, j-1, k)
         - PSGridGet(p0, i-1, j+1, k) + PSGridGet(p0, i-1, j-1, k) )
      + PSGridGet(b2, i, j, k)
      *( PSGridGet(p0, i+1, j, k+1) - PSGridGet(p0, i+1, j, k-1)
         - PSGridGet(p0, i-1, j, k+1) + PSGridGet(p0, i-1, j, k-1) )
      + PSGridGet(c0, i, j, k) * PSGridGet(p0, i, j, k-1)
      + PSGridGet(c1, i, j, k) * PSGridGet(p0, i, j-1, k)
      + PSGridGet(c2, i, j, k) * PSGridGet(p0, i-1, j, k)
      + PSGridGet(wrk1, i, j, k);
  ss = (s0 * PSGridGet(a3, i, j, k) - PSGridGet(p0, i, j, k))
       * PSGridGet(bnd, i, j, k);
  float v = PSGridGet(p0, i, j, k) + omega * ss;
  PSGridEmit(p1, v);
  return;
}

float
jacobi(int nn, PSGrid3DFloat a0, PSGrid3DFloat a1, PSGrid3DFloat a2,
       PSGrid3DFloat a3, PSGrid3DFloat b0, PSGrid3DFloat b1,
       PSGrid3DFloat b2,
       PSGrid3DFloat c0, PSGrid3DFloat c1, PSGrid3DFloat c2,
       PSGrid3DFloat p0, PSGrid3DFloat p1,
       PSGrid3DFloat bnd, PSGrid3DFloat wrk1)
{
  float  gosa = 0.0f;
  PSDomain3D innerDom = PSDomain3DNew(1, PSGridDim(p0, 0)-1,
                                      1, PSGridDim(p0, 1)-1,
                                      1, PSGridDim(p0, 2)-1);
  
  assert(nn % 2 == 0);
  
  printf("Executing jacobi\n");
  PSStencilRun(PSStencilMap(jacobi_kernel, innerDom,
                            p0, p1, a0, a1, a2, a3, b0, b1, b2,
                            c0, c1, c2, bnd, wrk1, omega),
               PSStencilMap(jacobi_kernel, innerDom,
                            p1, p0, a0, a1, a2, a3, b0, b1, b2,
                            c0, c1, c2, bnd, wrk1, omega),
               nn/2);
               
               /* PSStencilRun(PSStencilMap(gosa_kernel, innerDom, */
  /*                           wrk1, p0, a0, a1, a2, a3, */
  /*                           b0, b1, b2, c0, c1, c2, bnd, wrk1, omega)); */
  
  //gosa = grid_reduce(innerDom, wrk1, float_sum);
  return gosa;
}

double
second()
{
  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);
  
  if(base_sec == 0 && base_usec == 0)
  {
    base_sec = tm.tv_sec;
    base_usec = tm.tv_usec;
    t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) + 
        ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}

double GetThroughput(int nn, int mx, int my, int mz, double time)
{
  // load from a0-3, b0-2, c0-2, wrk1, bnd
  double load_interior_only =
      (double)(mz-2)*(double)(my-2)*(double)(mx-2)*12;
  // load from p0
  double load_whole =
      (double)(mz)*(double)(my)*(double)(mx)*1;
  // store to p1
  double store =
      (double)(mz-2)*(double)(my-2)*(double)(mx-2)*1;
  double total_size_per_update =
      sizeof(float) * (load_interior_only + load_whole + store);
  return (total_size_per_update * nn / time) * 1.0e-09;
}
