/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc.
 * References:
 * The Minimum Spanning Tree Problem: see SOM p 1469 */

#include <time.h>
#include <stdlib.h>
#include <libc.h>
#include "csv.h"
#include "vec.h"
#include "etc.h"
#include "som.h"

static Vec** load(int P, const char* file) {
  Csv* csv = newCsv(file);
  loadCsv(csv);
  Vec** pp = malloc(P * sizeof(Vec*));
  for (int p = 0; p < P; p++) {
    pp[p] = newVec(csv->F);
    for (int j = 0; j < csv->F; j++) pp[p]->c[j] = atof(csv->r[p][j]);
  }
  delCsv(csv);
  return pp;
}

static void toss(int P, Vec** pp) {
  for (int p = 0; p < P; p++) delVec(pp[p]);
  free(pp);
}

static Arch arch(const char* a) {
  if (strcmp(a, "r4") == 0) return r4;
  else if (strcmp(a, "r8") == 0) return r8;
  else {
    fprintf(stderr, "ERROR: unknown architecture %s\n", a);
    exit(1);
  }
}

static Dist dist(const char* d) {
  if (strcmp(d, "inner") == 0) return dotVVS;
  else if (strcmp(d, "euclidean") == 0) return eucVVS;
  else {
    fprintf(stderr, "ERROR: unknown distance measure %s\n", d);
    exit(1);
  }
}

static void run(const char* name) {
  // initialize
  char cwd[1024];
  getcwd(cwd, sizeof(cwd)); // current working directory
  char buf[1024];
  sprintf(buf, "%s/dat/%s.csv", cwd, name); // ~cwd/dat/"name".csv
  Csv* cfgcsv = newCsv(buf);
  loadCsv(cfgcsv);
  int f = 1; // CSV field; start at 1 to skip network name field
  int C = atoi(cfgcsv->r[1][f++]); // cfgcsv->r[1] holds data, r[0] holds header
  int I = atoi(cfgcsv->r[1][f++]);
  int W = atoi(cfgcsv->r[1][f++]);
  int H = atoi(cfgcsv->r[1][f++]);
  Arch a = arch(cfgcsv->r[1][f++]);
  Dist d = dist(cfgcsv->r[1][f++]);
  double alpha = atof(cfgcsv->r[1][f++]);
  double epsilon = atof(cfgcsv->r[1][f++]);
  int P = atoi(cfgcsv->r[1][f++]);
  bool shuffle = istrue(cfgcsv->r[1][f++]);
  delCsv(cfgcsv);
  cfgcsv = NULL;
  // load pattern vectors
  sprintf(buf, "%s/dat/%s-i.csv", cwd, name);
  Vec** ii = load(P, buf);
  // train network
  Som* som = newSom(name, alpha, epsilon, C, P, shuffle, I, H, W, a, d);
  learn(ii, som);
  dump(som);
  delSom(som);
  // terminate
  toss(P, ii);
}

int main(int argc, const char** argv) {
  srandom(time(NULL));
  if (argc != 2) {
    fprintf(stderr, "Usage: %s netname\n", argv[0]);
    exit(1);
  }
  const int T = 3; // number of trials
  for (int t = 0; t < T; t++) {
    printf("\n---- t = %d ----\n", t);
    run(argv[1]);
  }
  return 0;
}
