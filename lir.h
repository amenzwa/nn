/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc. */

#ifndef NN_LIR_H
#define NN_LIR_H

#include "etc.h"

extern double rampb(double);
extern double drampb(double);
extern double rampu(double);
extern double drampu(double);
extern double logisticb(double);
extern double dlogisticb(double);
extern double logisticu(double);
extern double dlogisticu(double);
extern double stepb(double);
extern double dstepb(double);
extern double stepu(double);
extern double dstepu(double);

typedef struct Bp {
  char* name; // network name
  double eta; // learning rate
  double alpha; // momentum factor
  double epsilon; // error criterion
  double e; // current cycle's error
  int L; // number of layers
  int I; // number of input taps
  int* N; // number of nodes N[l]
  Act* f; // activation function f[l]
  Act* df; // derivative of activation function df[l]
  double* p; // augmented input pattern
  double** i; // augmented input vector i[l][j]; pointers i[l] -> o[l-1]
  double** o; // augmented output vector o[l][j]
  double** d; // delta vector d[l][j]
  double*** w; // augmented weight matrix w[l][j][i]
  double*** dw; // augmented del-weight matrix dw[l][j][i]
} Bp;

extern Bp* newBp(const char* name, double eta, double alpha, double epsilon, int L, int I, const int* N, char** act);
extern void delBp(Bp* bp);
extern void learn(int C, int P, double** ii, double** tt, Bp* bp);
extern void recall(int P, double** ii, double** tt, Bp* bp);
extern void dump(Bp* bp);

#endif // NN_LIR_H
