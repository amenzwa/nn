/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc.
 * References:
 * SOM: The Self-Organizing Map, Kohonen (1990) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "etc.h"
#include "som.h"

static inline int side(Som* som) {
  /* Return the side length of the neighborhood square based on the current shrink. */
  return 1 + 2 * som->radius;
}

Som* newSom(const char* name, double alpha, double epsilon, int C, int P, bool shuffle, int I, int H, int W, Dist dist) {
  /* Create a network.
   * name: network name for use in report()
   * alpha: learning factor
   * epsilon: RMS error criterion
   * C: number of training cycles
   * P: number of pattern vectors
   * shuffle: shuffle the presentation order
   * I: number of input taps
   * H: height of the map
   * W: width of the map
   * dist: distance measure */
  Som* som = malloc(1 * sizeof(Som));
  som->name = strdup(name); // malloc()
  som->alpha = alpha;
  if (!(0.0 < som->alpha && som->alpha < 1.0)) { // see section II-B, SOM p 1467
    fprintf(stderr, "ERROR: alpha value %f is not within the open range (0.0, 1.0)\n", som->alpha);
    exit(1);
  }
  som->epsilon = epsilon;
  som->e = MAXFLOAT;
  som->C = C;
  som->P = P;
  som->shuffle = shuffle;
  som->ord = malloc(som->P * sizeof(int));
  for (int p = 0; p < som->P; p++) som->ord[p] = p;
  som->I = I;
  som->H = H;
  som->W = W;
  som->radius = som->W / 2; // see section II-D, SOM p 1469
  const int S = side(som);
  som->hood = malloc(S * S * sizeof(Loc)); // 1D array representing the 2D neighborhood square
  som->dist = dist;
  som->i = newVec(som->I);
  som->m = malloc(som->H * sizeof(Vec**));
  som->hits = malloc(som->H * sizeof(int*));
  for (int y = 0; y < som->H; y++) {
    som->m[y] = malloc(som->W * sizeof(Vec*));
    som->hits[y] = malloc(som->W * sizeof(int));
    for (int x = 0; x < som->W; x++) {
      som->m[y][x] = newVec(som->I);
      for (int i = 0; i < som->I; i++) som->m[y][x]->c[i] = randin(-WGT_RANGE / 2.0, +WGT_RANGE / 2.0); // symmetry breaking; see LIR p 10
      som->hits[y][x] = 0;
    }
  }
  return som;
}

void delSom(Som* som) {
  /* Destroy the network. */
  for (int y = 0; y < som->H; y++) {
    for (int x = 0; x < som->W; x++) delVec(som->m[y][x]);
    free(som->hits[y]);
    free(som->m[y]);
  }
  free(som->hits);
  free(som->m);
  delVec(som->i);
  free(som->hood);
  free(som->ord);
  free(som->name);
  free(som);
}

static Loc winner(const Vec* p, Som* som) {
  /* Select the winner node. */
  Loc n = {.x = -1, .y = -1}; // winner
  double min = MAXFLOAT;
  for (int y = 0; y < som->H; y++)
    for (int x = 0; x < som->W; x++) {
      double d = som->dist(p, som->m[y][x]);
      if (d < min) { // see eq 2', section II-B, SOM p 1467
        n = (Loc) {.x = x, .y = y};
        min = d;
      }
    }
  return n;
}

static inline int toindex(int w, int x, int y) {
  /* Convert (x, y) coordinate to 1D index. */
  return y * w + x;
}

static Loc* hood(int S, Loc n, Som* som) {
  /* Construct node n's neighborhood based on the current shrink.
   * See section II-B-D, SOM p 1467-1469 */
  Loc tl = (Loc) {.x = n.x - som->radius, .y = n.y - som->radius}; // top-left corner of the neighborhood
  for (int y = 0; y < S; y++)
    for (int x = 0; x < S; x++) som->hood[toindex(S, x, y)] = (Loc) {.x = tl.x + x, .y = tl.y + y};
  return som->hood;
}

static inline void shrink(int c, Som* som) {
  /* Monotonically shrink neighborhood radius after the ordering phase. */
  if (c >= ORDERING && som->radius > MIN_RADIUS) som->radius--; // see section II-D, SOM p 1469
}

static inline void slowdown(int C, int c, Som* som) {
  /* Monotonically decrease alpha after the ordering phase. */
  const double lim = 0.0001;
  if (c >= ORDERING && som->alpha > lim) {
    som->alpha -= (double) c / (double) C; // see section II-D, SOM p 1469
    if (som->alpha < lim) som->alpha = lim;
  }
}

static double alpha(Loc nc, Loc n, Som* som) {
  /* Return the alpha for a node in the neighborhood. */
  double d = (sqre(n.x - nc.x) + sqre(n.y - nc.y)) / sqre(som->radius); // scaled squared Euclidean distance
  return som->alpha * exp(-d); // see eq 8, section II-B, SOM p 1467
}

static void update(const Vec* x, Loc n, double a, Som* som) {
  Vec* w = som->m[n.y][n.x]; // [w] = [m]_winner
  subVVV(x, w, som->i); // [i] = [x] - [w]
  mulSVV(a, som->i, som->i); // [i] = alpha * [i]
  addVVV(w, som->i, w); // [w] = [w] + [i]; see eq 6, section II-B, SOM p 1467
}

static inline bool isinside(Loc n, Som* som) {
  return 0 <= n.y && n.y < som->H && 0 <= n.x && n.x < som->W;
}

static inline double sumsqre(double a, double c) {
  return a + sqre(c);
}

static inline void report(int c, Som* som) {
  printf("c = %-6d  e = %-10.8f\n", c, som->e);
}

void learn(Vec** ii, Som* som) {
  /* Train the network.
   * ii[]: input patterns */
  printf("learn %s\n", som->name);
  for (int c = 0; som->e > som->epsilon && c < som->C; c++) {
    if (som->shuffle) shuffle(som->P, som->ord);
    if (som->radius > MIN_RADIUS) shrink(c, som);
    slowdown(som->C, c, som);
    som->e = 0.0;
    for (int p = 0; p < som->P; p++) {
      // select the winner
      const Vec* v = ii[som->ord[p]];
      Loc nc = winner(v, som);
      som->hits[nc.y][nc.x]++; // update winner's hits
      // update weights of winner and its neighborhood
      const int S = side(som);
      Loc* hc = hood(S, nc, som);
      for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++) {
          Loc n = hc[toindex(S, x, y)];
          if (isinside(n, som)) update(v, n, alpha(nc, n, som), som);
        }
      som->e += foldVec(sumsqre, 0.0, som->i);
    }
    som->e = sqrt(som->e) / (som->W + som->H) / som->P;
    if (som->e < som->epsilon || c % (som->C / 10) == 0) report(c, som);
  }
}

void recall(Vec** ii, Som* som) {
  /* Test the network.
   * ii[]: input patterns */
  printf("recall %s\n", som->name);
  som->e = 0.0;
  for (int p = 0; p < som->P; p++) {
    // select the winner
    const Vec* v = ii[p];
    Loc nc = winner(v, som);
    printf("p = ");
    for (int i = 0; i < v->C; i++) printf("| %+10.4f ", v->c[i]);
    printf("| -> (%d, %d)\n", nc.x, nc.y);
    som->e += foldVec(sumsqre, 0.0, som->i);
  }
  som->e = sqrt(som->e) / (som->W + som->H) / som->P;
  report(-1, som);
}

void dump(Som* som) {
  printf("dump %s (%d x %d)\n", som->name, som->W, som->H);
  for (int y = 0; y < som->H; y++) {
    printf("  y = %d  ", y);
    for (int x = 0; x < som->W; x++) printf("| %8d ", som->hits[y][x]);
    printf("\n");
  }
}