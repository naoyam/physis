#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32

typedef struct {
  float r;
  float i;
} Complex;


#define OFFSET(x, y, z) ((((x)+N)%N) + (((y)+N)%N) * N + (((z)+N)%N) * N * N)

void kernel1(Complex *g1, Complex *g2) {
  int x, y, z;
  for (z = 0; z < N; ++z) {
    for (y = 0; y < N; ++y) {
      for (x = 0; x < N; ++x) {
        Complex t = g1[OFFSET(x, y, z)];
        Complex t1 = g1[OFFSET(x+1, y, z)];
        Complex t2 = g1[OFFSET(x-1, y, z)];
        Complex t3 = g1[OFFSET(x, y+1, z)];
        Complex t4 = g1[OFFSET(x, y-1, z)];
        Complex t5 = g1[OFFSET(x, y, z+1)];
        Complex t6 = g1[OFFSET(x, y, z-1)];
        float r = t.r + t1.r + t2.r + t3.r + t4.r + t5.r + t6.r;
        float i = t.i + t1.i + t2.i + t3.i + t4.i + t5.i + t6.i;        
        Complex v = {r, i};
        g2[OFFSET(x, y, z)] = v;
      }
    }
  }
  return;
}

void dump(Complex *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f\n", input[i].r, input[i].i);
  }
}

int main(int argc, char *argv[]) {
  Complex *g1, *g2;
  size_t nelms = N*N*N;
  g1 = (Complex *)malloc(sizeof(Complex) * nelms);
  g2 = (Complex *)malloc(sizeof(Complex) * nelms);  

  int i;
  for (i = 0; i < nelms; i++) {
    g1[i].r = i;
    g1[i].i = i+1;
  }

  kernel1(g1, g2);
  dump(g2);
  free(g1);
  free(g2);  
  return 0;
}

