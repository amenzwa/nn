/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#ifndef NN_LIR_H
#define NN_LIR_H

#include "etc.h"

typedef struct Ebp {
  char* name; // network name
  double eta; // learning rate
  double alpha; // momentum factor
  double epsilon; // error criterion
  double e; // current cycle's error
  int C; // number of training cycles
  int P; // number of data patterns
  bool shuffle; // shuffle input patterns
  int* ord; // input pattern presentation order
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
} Ebp;

extern Ebp* ebpnew(const char* name, double eta, double alpha, double epsilon, int C, int P, bool shuffle, int L, int I, const int* N, char** act);
extern void ebpdel(Ebp* ebp);
extern void learn(double** ii, double** tt, Ebp* ebp);
extern void recall(int P, double** ii, double** tt, Ebp* ebp);
extern void dump(Ebp* ebp);

#endif // NN_LIR_H
