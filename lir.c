/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc.
 * References:
 * LIR: Learning Internal Representations by Error Propagation, Rumelhart (1986)
 * ANS: Introduction to Artificial Neural Systems, Zurada (1992) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "csv.h"
#include "etc.h"
#include "lir.h"

void dump(const Ebp* ebp) {
  /* Dump the current weights. */
  printf("dump %s weights\n", ebp->name);
  for (int l = 0; l < ebp->L; l++) {
    printf("l = %d\n", l);
    for (int j = 0; j < ebp->N[l]; j++) {
      printf("  j = %d ", j);
      const int I = l == 0 ? ebp->I : ebp->N[l - 1];
      for (int i = 0; i <= I; i++) printf("| %+10.4f ", ebp->w[l][j][i]);
      printf("|\n");
    }
  }
}

inline void report(const Ebp* ebp, int c) {
  /* Report the current training cycle and current training error. */
  printf("c = %-10d  e = %-10.8f\n", c, ebp->e);
}

/* back-propagation */

Ebp* ebpnew(const char* name, double eta, double alpha, double epsilon, int nC, int nP, bool shuffle, int nL, int nI, const int* nN, char** act) {
  /* Create a network.
   * name: network name for use in report()
   * eta: learning rate
   * alpha: momentum factor
   * epsilon: RMS error criterion
   * nC: number of training cycles
   * nP: number of pattern vectors
   * shuffle: shuffle the presentation order
   * nL: number of processing layers
   * nI: number of input taps
   * nN[]: number of nodes per layer
   * act[]: name of activation function per layer */
  Ebp* ebp = malloc(sizeof(Ebp));
  ebp->name = strndup(name, FLDSIZ);  // malloc()
  ebp->eta = eta;
  ebp->alpha = alpha;
  ebp->epsilon = epsilon;
  ebp->e = DBL_MAX;
  ebp->C = nC;
  ebp->P = nP;
  ebp->shuffle = shuffle;
  ebp->order = malloc(ebp->P * sizeof(int));
  for (int p = 0; p < ebp->P; p++) ebp->order[p] = p;
  ebp->L = nL;
  ebp->I = nI;
  ebp->N = malloc(ebp->L * sizeof(int));
  ebp->f = malloc(ebp->L * sizeof(Act));
  ebp->df = malloc(ebp->L * sizeof(Act));
  ebp->p = malloc((ebp->I + 1) * sizeof(double)); // +1 augmentation for bias node; see fn 1, LIR p 9
  ebp->p[ebp->I] = 1.0;  // bias node output
  ebp->i = malloc(ebp->L * sizeof(double*));
  ebp->o = malloc(ebp->L * sizeof(double*));
  ebp->d = malloc(ebp->L * sizeof(double*));
  ebp->w = malloc(ebp->L * sizeof(double**));
  ebp->dw = malloc(ebp->L * sizeof(double**));
  for (int l = 0; l < ebp->L; l++) {
    const int J = nN[l];
    const int I = l == 0 ? ebp->I : nN[l - 1];
    ebp->N[l] = J;
    const ActPair p = actpair(act[l]);
    ebp->f[l] = p.f;
    ebp->df[l] = p.df;
    ebp->i[l] = l == 0 ? ebp->p : ebp->o[l - 1];  // point to upstream layer's augmented output vector
    ebp->o[l] = malloc((J + 1) * sizeof(double));
    ebp->o[l][J] = 1.0;  // bias node output
    ebp->d[l] = malloc(J * sizeof(double));
    ebp->w[l] = malloc(J * sizeof(double*));
    ebp->dw[l] = malloc(J * sizeof(double*));
    for (int j = 0; j < J; j++) {
      ebp->w[l][j] = malloc((I + 1) * sizeof(double));
      ebp->dw[l][j] = malloc((I + 1) * sizeof(double));
      for (int i = 0; i <= I; i++) {
        ebp->w[l][j][i] = randin(-WGT_RNG / 2.0, +WGT_RNG / 2.0); // symmetry breaking; see LIR p 10
        ebp->dw[l][j][i] = 0.0;
      }
    }
  }
  return ebp;
}

void ebpdel(Ebp* ebp) {
  /* Destroy the network. */
  for (int l = 0; l < ebp->L; l++) {
    for (int j = 0; j < ebp->N[l]; j++) {
      free(ebp->dw[l][j]);
      free(ebp->w[l][j]);
    }
    free(ebp->dw[l]);
    free(ebp->w[l]);
    free(ebp->d[l]);
    free(ebp->o[l]);
  }
  free(ebp->dw);
  ebp->dw = NULL;
  free(ebp->w);
  ebp->w = NULL;
  free(ebp->d);
  ebp->d = NULL;
  free(ebp->o);
  ebp->o = NULL;
  free(ebp->i);
  ebp->i = NULL;
  free(ebp->p);
  ebp->p = NULL;
  free(ebp->df);
  ebp->df = NULL;
  free(ebp->f);
  ebp->f = NULL;
  free(ebp->N);
  ebp->N = NULL;
  free(ebp->order);
  ebp->order = NULL;
  free(ebp->name);
  ebp->name = NULL;
  free(ebp);
}

static void forward(Ebp* ebp, const double* p) {
  /* Feed the pattern p forward. */
  memcpy(ebp->p, p, ebp->I * sizeof(double)); // network [p] = input [p]; does not overwrite bias node
  // feed forward pattern
  for (int l = 0; l < ebp->L; l++) { // from the first layer to the last
    const int J = ebp->N[l];
    for (int j = 0; j < J; j++) {
      const int I = l == 0 ? ebp->I : ebp->N[l - 1];
      double net = 0.0;
      for (int i = 0; i <= I; i++) net += ebp->w[l][j][i] * ebp->i[l][i];
      ebp->o[l][j] = ebp->f[l](net); // see eq 7, LIR p 6
    }
  }
}

static void backward(Ebp* ebp, const double* p) {
  const int lo = ebp->L - 1;
  for (int l = lo; l >= 0; l--) { // from the last layer to the first
    const int J = ebp->N[l];
    // calculate deltas
    if (l == lo) { // for output nodes
      for (int j = 0; j < J; j++) {
        double err = p[j] - ebp->o[l][j];
        ebp->d[l][j] = err * ebp->df[l](ebp->o[l][j]); // see eq 13, LIR p 7
      }
    } else { // for hidden nodes
      const int ld = l + 1; // adjacent downstream layer
      const int K = ebp->N[ld];
      for (int j = 0; j < J; j++) {
        double err = 0.0;
        for (int k = 0; k < K; k++) err += ebp->w[ld][k][j] * ebp->d[ld][k];
        ebp->d[l][j] = err * ebp->df[l](ebp->o[l][j]); // see eq 14, LIR p 7
      }
    }
    // calculate del-weights
    for (int j = 0; j < ebp->N[l]; j++) {
      const int I = l == 0 ? ebp->I : ebp->N[l - 1];
      for (int i = 0; i <= I; i++) ebp->dw[l][j][i] = ebp->eta * ebp->d[l][j] * ebp->i[l][i] + ebp->alpha * ebp->dw[l][j][i]; // see eq 16, LIR p 9
    }
  }
}

void learn(Ebp* ebp, double** ii, double** tt) {
  /* Train the network.
   * ii[]: input patterns
   * tt[]: associated target patterns (to calculate recall errors) */
  printf("learn %s\n", ebp->name);
  const int lo = ebp->L - 1;
  for (int c = 0; ebp->e > ebp->epsilon && c < ebp->C; c++) {
    // learn one cycle
    if (ebp->shuffle) shuffle(ebp->P, ebp->order);
    ebp->e = 0.0;
    for (int p = 0; p < ebp->P; p++) {
      forward(ebp, ii[ebp->order[p]]);
      backward(ebp, tt[ebp->order[p]]);
      for (int j = 0; j < ebp->N[lo]; j++) ebp->e += sqre(ebp->d[lo][j]); // sum of squares error; see LIR p 4
    }
    // update weights at end of cycle
    for (int l = 0; l < ebp->L; l++)
      for (int j = 0; j < ebp->N[l]; j++) {
        const int I = l == 0 ? ebp->I : ebp->N[l - 1];
        for (int i = 0; i <= I; i++) ebp->w[l][j][i] += ebp->dw[l][j][i]; // (w) = (w) + (dw)
      }
    // report training error
    ebp->e = sqrt(ebp->e) / ebp->N[lo] / ebp->P; // root-mean-square error; see eq 4.35, ANS p 196
    if (ebp->e < ebp->epsilon || c % (ebp->C / 10) == 0) report(ebp, c);
  }
}

void recall(Ebp* ebp, int P, double** ii, double** tt) {
  /* Test the network.
   * P: number of data patterns
   * ii[]: input patterns
   * tt[]: associated target patterns (to calculate recall errors) */
  printf("recall %s\n", ebp->name);
  const int lo = ebp->L - 1;
  ebp->e = 0.0;
  for (int p = 0; p < P; p++) {
    // feed a test pattern
    const double* i = ii[p];
    const double* t = tt[p];
    forward(ebp, i);
    for (int j = 0; j < ebp->N[lo]; j++) ebp->e += sqre(t[j] - ebp->o[lo][j]);
    // show input-output associations
    printf("p = %-10d\n", p);
    printf("  i = ");
    for (int j = 0; j < ebp->I; j++) printf("| %+10.4f ", i[j]);
    printf("|\n  o = ");
    for (int j = 0; j < ebp->N[lo]; j++) printf("| %+10.4f ", ebp->o[lo][j]);
    printf("|\n  t = ");
    for (int j = 0; j < ebp->N[lo]; j++) printf("| %+10.4f ", t[j]);
    printf("|\n");
  }
  // report recall error
  ebp->e = sqrt(ebp->e) / ebp->N[lo] / P;
  report(ebp, -1);
}