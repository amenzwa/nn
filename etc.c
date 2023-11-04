/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "etc.h"

inline bool iszero(double x) {
  /* Due to the imprecision of the FPU's representation of real numbers,
   * it is impossible to check for exact zero equality, as in x == 0.0. */
  return fabs(x) <= 1.0e-12;
}

inline bool istrue(const char* s) {
  return strcasecmp(s, "true") == 0;
}

inline double sqre(double x) {
  return x * x;
}

inline double sumsqre(double a, double c) {
  return a + sqre(c);
}

inline double randin(double lo, double hi) {
  /* Return a random double in the range [lo, hi]. */
  return lo + (double) random() / (double) RAND_MAX * (hi - lo);
}

void shuffle(int N, int* ord) {
  for (int i = 0; i < N - 1; i++) {
    int j = (int) (i + random() / (RAND_MAX / (N - i) + 1));
    int t = ord[j];
    ord[j] = ord[i];
    ord[i] = t;
  }
}

/* activation functions */

inline double ramp(double x) {
  return x;
}

inline double dramp(double /*x*/) {
  return 1.0;
}

inline double relu(double x) {
  return (x > 0.0 ? 1.0 : 0.01) * x; // rectified linear unit activation function
}

inline double drelu(double x) {
  return x > 0.0 ? 1.0 : 0.01; // derivative of rectified linear unit activation function
}

inline double logisticb(double x) {
  return 2.0 / (1.0 + exp(-x)) - 1.0; // bipolar logistic activation function; see ANS p 180
}

inline double dlogisticb(double x) {
  return 0.5 * (1.0 - sqre(x)); // derivative of bipolar logistic activation function; see eq 4.18b, ANS p 179
}

inline double logisticu(double x) {
  return 1.0 / (1.0 + exp(-x)); // unipolar logistic activation function; see eq 15, LIR p 9
}

inline double dlogisticu(double x) {
  return x - sqre(x); // derivative of unipolar logistic activation function; see LIR p 9
}

inline double stepb(double x) {
  return x < 0.0 ? -0.99 : (x > 0.0 ? 0.99 : 0.0);
}

inline double dstepb(double x) {
  return iszero(x) ? 1.99 : 0.01;
}

inline double stepu(double x) {
  return x < 0.0 ? 0.01 : (x > 0.0 ? 0.99 : 0.0);
}

inline double dstepu(double x) {
  return iszero(x) ? 0.99 : 0.01;
}

ActPair actpair(const char* act) {
  if (strcmp(act, "ramp") == 0) return (ActPair) {.f = ramp, .df = dramp};
  else if (strcmp(act, "relu") == 0) return (ActPair) {.f = relu, .df = drelu};
  else if (strcmp(act, "logisticb") == 0) return (ActPair) {.f = logisticb, .df = dlogisticb};
  else if (strcmp(act, "logisticu") == 0) return (ActPair) {.f = logisticu, .df = dlogisticu};
  else if (strcmp(act, "stepb") == 0) return (ActPair) {.f = stepb, .df = dstepb};
  else if (strcmp(act, "stepu") == 0) return (ActPair) {.f = stepu, .df = dstepu};
  fprintf(stderr, "ERROR: unknown activation function\n");
  exit(1);
}