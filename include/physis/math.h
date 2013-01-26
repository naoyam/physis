#ifndef PHYSIS_MATH_H_
#define PHYSIS_MATH_H_

#ifdef PHYSIS_USER
extern double exp(double x);
extern float expf(float x);
extern long double expl(long double x);
extern double cos(double x);
extern float cosf(float x);
extern double acos(double x);
extern float acosf(float x);
#else
#if defined(PHYSIS_REF) || defined(PHYSIS_MPI) || defined(PHYSIS_MPI_OPENMP)
#include <math.h>
#endif
#endif



#endif /* PHYSIS_MATH_H_ */
