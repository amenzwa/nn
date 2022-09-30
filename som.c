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

Som* newSom(const char* name, double alpha, double epsilon, int C, int P, bool shuffle, int I, int H, int W, Arch arch, Dist dist) {
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
  som->arch = arch;
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
  for (int y = 0; y < som->H; y++) {
    for (int x = 0; x < som->W; x++) delVec(som->m[y][x]);
    free(som->hits[y]);
    free(som->m[y]);
  }
  free(som->hits);
  free(som->m);
  delVec(som->i);
  free(som->ord);
  free(som->name);
  free(som);
}

static double alpha(int C, int c, Som* som) {
  /* Monotinically decrease alpha after the ordering phase. */
  if (c >= ORDERING) som->alpha -= (som->alpha - 0.1) / (double) (C - ORDERING); // see section II-D, SOM p 1469
  return som->alpha;
}

static Node winner(const Vec* p, Som* som) {
  /* Select the winner node. */
  Node n = {.x = -1, .y = -1}; // winner
  double min = MAXFLOAT;
  for (int y = 0; y < som->H; y++)
    for (int x = 0; x < som->W; x++) {
      double d = som->dist(p, som->m[y][x]);
      if (d < min) { // see eq 2', section II-B, SOM p 1467
        n = (Node) {.x = x, .y = y};
        min = d;
      }
    }
  return n;
}

static const Node* hood(int /*c*/, Node n, Som* som) {
  /* Return n's neighborhood.
   * See section II-B-D, SOM p 1467-1469 */
  const int radius = 1; // TODO start with radius of W/2 and shrink incrementally down to 1 after c has gone past ORDERING
  // neighborhood for r4
  som->hood[north] = (Node) {.x = n.x, .y = n.y - radius};
  som->hood[east] = (Node) {.x = n.x + radius, .y = n.y};
  som->hood[south] = (Node) {.x = n.x, .y = n.y + radius};
  som->hood[west] = (Node) {.x = n.x - radius, .y = n.y};
  // neighborhood for r8
  if (som->arch == r8) {
    som->hood[northeast] = (Node) {.x = n.x + radius, .y = n.y - radius};
    som->hood[southeast] = (Node) {.x = n.x + radius, .y = n.y + radius};
    som->hood[southwest] = (Node) {.x = n.x - radius, .y = n.y + radius};
    som->hood[northwest] = (Node) {.x = n.x - radius, .y = n.y - radius};
  }
  return som->hood;
}

static double sumsqre(double a, double c) {
  return a + sqre(c);
}

static void update(const Vec* x, Node n, double a, Som* som) {
  Vec* w = som->m[n.y][n.x]; // [w] = [m]_winner
  subVVV(x, w, som->i); // [i] = [x] - [w]
  mulSVV(a, som->i, som->i); // [i] = alpha * [i]
  addVVV(w, som->i, w); // [w] = [w] + [i]; see eq 6, section II-B, SOM p 1467
  som->e += foldVec(sumsqre, 0.0, som->i);
}

static bool isinside(Node n, Som* som) {
  return 0 <= n.y && n.y < som->H && 0 <= n.x && n.x < som->W;
}

static void report(int c, Som* som) {
  printf("c = %-6d  e = %f\n", c, som->e);
}

void learn(Vec** ii, Som* som) {
  /* Train the network.
   * ii[]: input patterns */
  printf("learn %s\n", som->name);
  for (int c = 0; som->e > som->epsilon && c < som->C; c++) {
    if (som->shuffle) shuffle(som->P, som->ord);
    som->e = 0.0;
    for (int p = 0; p < som->P; p++) {
      // select the winner
      const Vec* x = ii[som->ord[p]];
      Node nc = winner(x, som);
      som->hits[nc.y][nc.x]++; // update hit count
      // update weights of winner and its neighborhood
      const double a = alpha(som->C, c, som);
      update(x, nc, a, som);
      const Node* h = hood(c, nc, som);
      for (int n = 0; n < som->arch; n++)
        if (isinside(h[n], som)) update(x, h[n], 0.5 * a, som); // use 0.5 alpha for neighbors
    }
    som->e = sqrt(som->e) / (som->W + som->H) / som->P;
    if (som->e < som->epsilon || c % (som->C / 10) == 0) report(c, som);
  }
}

void dump(Som* som) {
  printf("dump %s (%d x %d)\n", som->name, som->W, som->H);
  for (int y = 0; y < som->H; y++) {
    for (int x = 0; x < som->W; x++) printf(" %8d", som->hits[y][x]);
    printf("\n");
  }
}