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

static inline bool isinside(Som* som, Loc n) {
  /* Check if node n is inside the map. */
  return 0 <= n.y && n.y < som->H && 0 <= n.x && n.x < som->W;
}

static inline bool isordering(int c) {
  /* Check if the learning process is still in the ordering phase. */
  return c < ORDERING; // see section II-D, SOM p 1469
}

static inline int toindex(int w, int x, int y) {
  /* Convert (x, y) coordinate to 1D index. */
  return y * w + x;
}

static inline int radius(Som* som, int c) {
  /* Monotonically shrink neighborhood radius after the ordering phase. */
  if (isordering(c)) return som->radius;
  const int r = (int) (som->radius * exp(-(double) c / som->C)); // see section II-D, SOM p 1469
  return r <= MIN_RADIUS ? MIN_RADIUS : r;
}

static inline int side(Som* som, int c) {
  /* Return the side length of the neighborhood square based on the current shrink. */
  return 1 + 2 * radius(som, c);
}

static Loc* hood(Som* som, int c, int S, Loc n) {
  /* Construct node n's neighborhood based on the current radius.
   * See section II-B-D, SOM p 1467-1469 */
  const int r = radius(som, c);
  Loc tl = (Loc) {.x = n.x - r, .y = n.y - r}; // top-left corner of the neighborhood
  for (int y = 0; y < S; y++)
    for (int x = 0; x < S; x++) som->hood[toindex(S, x, y)] = (Loc) {.x = tl.x + x, .y = tl.y + y};
  return som->hood;
}

static double alpha(Som* som, int c, Loc nc, Loc n) {
  /* Monotonically decrease alpha after the ordering phase, and return the alpha for a node in the neighborhood. */
  if (isordering(c)) return som->alpha;
  const double d = (sqre(n.x - nc.x) + sqre(n.y - nc.y)) / sqre(radius(som, c)); // scaled squared Euclidean distance
  const double a = som->alpha * exp(-d - (double) c / som->C); // see eq 8, section II-B, SOM p 1467
  return a <= END_ALPHA ? END_ALPHA : a;
}

void dump(Som* som) {
  /* Dump the current hits. */
  printf("dump %s (%d x %d) hits\n", som->name, som->W, som->H);
  for (int y = 0; y < som->H; y++) {
    printf("  y = %d  ", y);
    for (int x = 0; x < som->W; x++) printf("| %8d ", som->hits[y][x]);
    printf("\n");
  }
}

static inline void report(Som* som, int c) {
  /* Report the current training cycle and current training error. */
  printf("c = %-10d  e = %-10.8f\n", c, som->e);
}

/* self-organizing map */

Som* somnew(const char* name, double alpha, double epsilon, int C, int P, bool shuffle, int I, int H, int W, Dist dist) {
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
  const int S = side(som, 0);
  som->hood = malloc(S * S * sizeof(Loc)); // 1D array representing the 2D neighborhood square
  som->dist = dist;
  som->i = vecnew(som->I);
  som->m = malloc(som->H * sizeof(Vec**));
  som->hits = malloc(som->H * sizeof(int*));
  for (int y = 0; y < som->H; y++) {
    som->m[y] = malloc(som->W * sizeof(Vec*));
    som->hits[y] = malloc(som->W * sizeof(int));
    for (int x = 0; x < som->W; x++) {
      som->m[y][x] = vecnew(som->I);
      for (int i = 0; i < som->I; i++) som->m[y][x]->c[i] = randin(-WGT_RANGE / 2.0, +WGT_RANGE / 2.0); // symmetry breaking; see LIR p 10
      som->hits[y][x] = 0;
    }
  }
  return som;
}

void somdel(Som* som) {
  /* Destroy the network. */
  for (int y = 0; y < som->H; y++) {
    for (int x = 0; x < som->W; x++) vecdel(som->m[y][x]);
    free(som->hits[y]);
    free(som->m[y]);
  }
  free(som->hits);
  free(som->m);
  vecdel(som->i);
  free(som->hood);
  free(som->ord);
  free(som->name);
  free(som);
}

static Loc winner(Som* som, const Vec* p) {
  /* Select the winner. */
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

static void update(Som* som, const Vec* x, Loc n, double a) {
  /* Update the weights of the winner and its neighborhood. */
  Vec* w = som->m[n.y][n.x]; // [w] = [m]_winner
  vecsub(som->i, x, w); // [i] = [x] - [w]
  vecscale(som->i, a, som->i); // [i] = alpha * [i]
  vecadd(w, w, som->i); // [w] = [w] + [i]; see eq 6, section II-B, SOM p 1467
}

void learn(Som* som, Vec** ii) {
  /* Train the network.
   * ii[]: input patterns */
  printf("learn %s\n", som->name);
  for (int c = 0; som->e > som->epsilon && c < som->C; c++) {
    som->e = 0.0;
    if (som->shuffle) shuffle(som->P, som->ord);
    for (int p = 0; p < som->P; p++) {
      // select the winner
      const Vec* v = ii[som->ord[p]];
      Loc nc = winner(som, v);
      som->hits[nc.y][nc.x]++; // update winner's hits
      // update weights of winner and its neighborhood
      const int S = side(som, c);
      Loc* hc = hood(som, c, S, nc);
      for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++) {
          Loc n = hc[toindex(S, x, y)];
          if (isinside(som, n)) update(som, v, n, alpha(som, c, nc, n));
        }
      som->e += vecfold(sumsqre, 0.0, som->i);
    }
    // report training error
    som->e = sqrt(som->e) / (som->W + som->H) / som->P;
    if (som->e < som->epsilon || c % (som->C / 10) == 0) report(som, c);
  }
}

void recall(Som* som, Vec** ii) {
  /* Test the network.
   * ii[]: input patterns */
  printf("recall %s\n", som->name);
  som->e = 0.0;
  for (int p = 0; p < som->P; p++) {
    // select the winner
    const Vec* v = ii[p];
    Loc nc = winner(som, v);
    som->e += vecfold(sumsqre, 0.0, som->i);
    // show pattern-winner association
    printf("p = %-10d ", p);
    for (int i = 0; i < v->C; i++) printf("| %+10.4f ", v->c[i]);
    printf("| -> (%d, %d)\n", nc.x, nc.y);
  }
  // report recall error
  som->e = sqrt(som->e) / (som->W + som->H) / som->P;
  report(som, -1);
}