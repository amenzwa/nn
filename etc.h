/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc. */

#ifndef NN_ETC_H
#define NN_ETC_H

#include <stdbool.h>

#define WGT_RANGE 0.5  // weight initialization range

typedef double (* Act)(double); // activation function

extern bool iszero(double x);
extern double sqre(double x);
extern double randin(double lo, double hi);

extern double rampb(double x);
extern double drampb(double);
extern double rampu(double x);
extern double drampu(double x);
extern double logisticb(double x);
extern double dlogisticb(double x);
extern double logisticu(double x);
extern double dlogisticu(double x);
extern double stepb(double x);
extern double dstepb(double x);
extern double stepu(double x);
extern double dstepu(double x);
extern void setact(const char* act, Act* f, Act* df);

#endif // NN_ETC_H
