/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#ifndef NN_SOM_H
#define NN_SOM_H

#include <stdbool.h>
#include "vec.h"

#define ORDERING 1000 // number of cycles for early, ordering phase
#define RADIUS_MIN 1 // minimum neighborhood radius
#define ALPHA_MIN 0.1 // ending learning factor

typedef struct Loc {
  int x, y; // node location on the map
} Loc;

typedef double (* Dist)(const Vec* u, const Vec* v);

typedef struct Som {
  char* name; // network name
  double alpha; // beginning learning factor
  double epsilon; // error criterion
  double e; // current cycle's error
  int C; // number of training cycles
  int P; // number of data patterns
  bool shuffle; // shuffle the input vectors
  int* ord; // input presentation order
  int I; // input vector length
  int H, W; // network dimensions
  int radius; // beginning neighborhood radius
  Loc* hood; // neighborhood around the winner
  Dist dist; // distance measure
  Vec* i; // temporary store for alpha * [x]
  Vec*** m; // grid of code vectors m[r][c] = w[i]
  int** hits; // hits per node
} Som;

extern Som* somnew(const char* name, double alpha, double epsilon, int C, int P, bool shuffle, int I, int H, int W, Dist dist);
extern void somdel(Som* som);
extern void learn(Som* som, Vec** ii);
extern void recall(Som* som, Vec** ii);
extern void dump(Som* som);

#endif // NN_SOM_H
