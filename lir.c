/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc.
 * References:
 * LIR: Learning Internal Representations by Error Propagation, Rumelhart (1986)
 * ANS: Introduction to Artificial Neural Systems, Zurada (1992) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "etc.h"
#include "lir.h"

Bp* newBp(const char* name, double eta, double alpha, double epsilon, int nL, int nI, const int* nN, char** act) {
  /* Create a new network.
   * name: network name for use in report()
   * eta: learning rate
   * alpha: momentum factor
   * epsilon: RMS error criterion
   * nL: number of processing layers
   * nI: number of input taps
   * nN[]: number of nodes per layer
   * act[]: name of activation function per layer */
  Bp* bp = malloc(1 * sizeof(Bp));
  bp->name = strdup(name);  // malloc()
  bp->eta = eta;
  bp->alpha = alpha;
  bp->epsilon = epsilon;
  bp->e = MAXFLOAT;
  bp->L = nL;
  bp->I = nI;
  bp->N = malloc(bp->L * sizeof(int));
  bp->f = malloc(bp->L * sizeof(Act));
  bp->df = malloc(bp->L * sizeof(Act));
  bp->p = malloc((bp->I + 1) * sizeof(double)); // +1 augmentation for bias node; see fn 1, LIR p 9
  bp->p[bp->I] = 1.0;  // bias node output
  bp->i = malloc(bp->L * sizeof(double*));
  bp->o = malloc(bp->L * sizeof(double*));
  bp->d = malloc(bp->L * sizeof(double*));
  bp->w = malloc(bp->L * sizeof(double**));
  bp->dw = malloc(bp->L * sizeof(double**));
  for (int l = 0; l < bp->L; l++) {
    const int J = nN[l];
    const int I = l == 0 ? bp->I : nN[l - 1];
    bp->N[l] = J;
    setact(act[l], &(bp->f[l]), &(bp->df[l]));
    bp->i[l] = l == 0 ? bp->p : bp->o[l - 1];  // point to upstream layer's augmented output vector
    bp->o[l] = malloc((J + 1) * sizeof(double));
    bp->o[l][J] = 1.0;  // bias node output
    bp->d[l] = malloc(J * sizeof(double));
    bp->w[l] = malloc(J * sizeof(double*));
    bp->dw[l] = malloc(J * sizeof(double*));
    for (int j = 0; j < J; j++) {
      bp->w[l][j] = malloc((I + 1) * sizeof(double));
      bp->dw[l][j] = malloc((I + 1) * sizeof(double));
      for (int i = 0; i <= I; i++) {
        bp->w[l][j][i] = randin(-WGT_RANGE / 2.0, +WGT_RANGE / 2.0); // symmetry breaking; see LIR p 10
        bp->dw[l][j][i] = 0.0;
      }
    }
  }
  return bp;
}

void delBp(Bp* bp) {
  /* Destroy the network. */
  for (int l = 0; l < bp->L; l++) {
    for (int j = 0; j < bp->N[l]; j++) {
      free(bp->dw[l][j]);
      free(bp->w[l][j]);
    }
    free(bp->dw[l]);
    free(bp->w[l]);
    free(bp->d[l]);
    free(bp->o[l]);
  }
  free(bp->dw);
  free(bp->w);
  free(bp->d);
  free(bp->o);
  free(bp->i);
  free(bp->p);
  free(bp->df);
  free(bp->f);
  free(bp->N);
  free(bp->name);
  free(bp);
}

static void forward(const double* p, Bp* bp) {
  /* Feed the pattern p forward. */
  memcpy(bp->p, p, bp->I * sizeof(double)); // network [p] = input [p]; does not overwrite bias node
  // feed forward pattern
  for (int l = 0; l < bp->L; l++) { // from the first layer to the last
    const int J = bp->N[l];
    for (int j = 0; j < J; j++) {
      const int I = l == 0 ? bp->I : bp->N[l - 1];
      double net = 0.0;
      for (int i = 0; i <= I; i++) net += bp->w[l][j][i] * bp->i[l][i];
      bp->o[l][j] = bp->f[l](net); // see eq 7, LIR p 6
    }
  }
}

static void backward(const double* p, Bp* bp) {
  const int lo = bp->L - 1;
  for (int l = lo; l >= 0; l--) { // from the last layer to the first
    const int J = bp->N[l];
    // calculate deltas
    if (l == lo) { // for output nodes
      for (int j = 0; j < J; j++) {
        double err = p[j] - bp->o[l][j];
        bp->d[l][j] = err * bp->df[l](bp->o[l][j]); // see eq 13, LIR p 7
      }
    } else { // for hidden nodes
      const int ld = l + 1; // adjacent downstream layer
      const int K = bp->N[ld];
      for (int j = 0; j < J; j++) {
        double err = 0.0;
        for (int k = 0; k < K; k++) err += bp->w[ld][k][j] * bp->d[ld][k];
        bp->d[l][j] = err * bp->df[l](bp->o[l][j]); // see eq 14, LIR p 7
      }
    }
    // calculate del-weights
    for (int j = 0; j < bp->N[l]; j++) {
      const int I = l == 0 ? bp->I : bp->N[l - 1];
      for (int i = 0; i <= I; i++) bp->dw[l][j][i] = bp->eta * bp->d[l][j] * bp->i[l][i] + bp->alpha * bp->dw[l][j][i]; // see eq 16, LIR p 9
    }
  }
}

static void report(int c, Bp* bp) {
  printf("c = %-6d  e = %-10.4g\n", c, bp->e);
}

void learn(int C, int P, double** ii, double** tt, Bp* bp) {
  /* Train the network.
   * C: number of training cycles
   * P: number of data patterns
   * ii[]: input patterns
   * tt[]: associated target patterns (to calculate recall errors) */
  printf("\nlearn %s\n", bp->name);
  const int lo = bp->L - 1;
  for (int c = 0; bp->e > bp->epsilon && c < C; c++) {
    // learn one cycle
    bp->e = 0.0;
    for (int p = 0; p < P; p++) {
      forward(ii[p], bp);
      backward(tt[p], bp);
      for (int j = 0; j < bp->N[lo]; j++) bp->e += sqre(bp->d[lo][j]); // sum of squares error; see LIR p 4
    }
    bp->e = sqrt(bp->e) / bp->N[lo] / P; // root-mean-square error; see eq 4.35, ANS p 196
    // update weights at end of cycle
    for (int l = 0; l < bp->L; l++)
      for (int j = 0; j < bp->N[l]; j++) {
        const int I = l == 0 ? bp->I : bp->N[l - 1];
        for (int i = 0; i <= I; i++) bp->w[l][j][i] += bp->dw[l][j][i]; // (w) = (w) + (dw)
      }
    if (bp->e < bp->epsilon || c % (C / 10) == 0) report(c, bp);
  }
}

void recall(int P, double** ii, double** tt, Bp* bp) {
  /* Test the network.
   * P: number of data patterns
   * ii[]: input patterns
   * tt[]: associated target patterns (to calculate recall errors) */
  printf("recall %s\n", bp->name);
  const int lo = bp->L - 1;
  bp->e = 0.0;
  for (int p = 0; p < P; p++) {
    forward(ii[p], bp);
    for (int j = 0; j < bp->N[lo]; j++) bp->e += sqre(tt[p][j] - bp->o[lo][j]);
  }
  bp->e = sqrt(bp->e) / bp->N[lo] / P;
  report(-1, bp);
}

void dump(Bp* bp) {
  printf("dump %s weights\n", bp->name);
  for (int l = 0; l < bp->L; l++) {
    printf("l = %d\n", l);
    for (int j = 0; j < bp->N[l]; j++) {
      printf("  j = %d ", j);
      const int I = l == 0 ? bp->I : bp->N[l - 1];
      for (int i = 0; i <= I; i++) printf("| %-10.4g ", bp->w[l][j][i]);
      printf("|\n");
    }
  }
}