/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#ifndef NN_ETC_H
#define NN_ETC_H

#include <stdbool.h>

#define WGT_RNG 0.6  // weight initialization range; see LIR p 31

typedef double (* Act)(double); // activation function
typedef struct ActPair {
  Act f, df;
} ActPair;

extern bool istrue(const char* s);
extern bool iszero(double x);
extern double sqre(double x);
extern double sumsqre(double a, double c);
extern double randin(double lo, double hi);
extern void shuffle(int N, int* ord);
extern double linear(double x);
extern double dlinear(double);
extern double relu(double x);
extern double drelu(double x);
extern double logisticb(double x);
extern double dlogisticb(double x);
extern double logisticu(double x);
extern double dlogisticu(double x);
extern double stepb(double x);
extern double dstepb(double x);
extern double stepu(double x);
extern double dstepu(double x);
extern ActPair actpair(const char* act);

#endif // NN_ETC_H
