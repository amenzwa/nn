/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc. */

#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "etc.h"

bool istrue(const char* s) {
  return strcasecmp(s, "true") == 0;
}

bool iszero(double x) {
  /* The imprecision in the FPU hardware representation of real numbers,
   * it is impossible to check for exact zero equality like x == 0.0. */
  return fabs(x) <= 10e-12;
}

double sqre(double x) {
  return x * x;
}

double randin(double lo, double hi) {
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

double rampb(double x) {
  return x;
}

double drampb(double /*x*/) {
  return 1.0;
}

double rampu(double x) {
  return x < 0.0 ? 0.01 : (x > 0.0 ? x : 0.0);
}

double drampu(double x) {
  return x < 0.0 ? 0.01 : (x > 0.0 ? 0.99 : 0.0);
}

double logisticb(double x) {
  return 2.0 / (1.0 + exp(-x)) - 1.0; // bipolar logistic activation function; see ANS p 180
}

double dlogisticb(double x) {
  return 0.5 * (1.0 - sqre(x)); // derivative of bipolar logistic activation function; see eq 4.18b, ANS p 179
}

double logisticu(double x) {
  return 1.0 / (1.0 + exp(-x)); // unipolar logistic activation function; see eq 15, LIR p 9
}

double dlogisticu(double x) {
  return x - sqre(x); // derivative of unipolar logistic activation function; see LIR p 9
}

double stepb(double x) {
  return x < 0.0 ? -0.99 : (x > 0.0 ? 0.99 : 0.0);
}

double dstepb(double x) {
  return iszero(x) ? 1.99 : 0.01;
}

double stepu(double x) {
  return x < 0.0 ? 0.01 : (x > 0.0 ? 0.99 : 0.0);
}

double dstepu(double x) {
  return iszero(x) ? 0.99 : 0.01;
}

void setact(const char* act, Act* f, Act* df) {
  if (strcmp(act, "rampb") == 0) {
    *f = rampb;
    *df = drampb;
  } else if (strcmp(act, "rampu") == 0) {
    *f = rampu;
    *df = drampu;
  } else if (strcmp(act, "logisticb") == 0) {
    *f = logisticb;
    *df = dlogisticb;
  } else if (strcmp(act, "logisticu") == 0) {
    *f = logisticu;
    *df = dlogisticu;
  } else if (strcmp(act, "stepb") == 0) {
    *f = stepb;
    *df = dstepb;
  } else if (strcmp(act, "stepu") == 0) {
    *f = stepu;
    *df = dstepu;
  } else {
    fprintf(stderr, "ERROR: unknown activation function\n");
    exit(1);
  }
}