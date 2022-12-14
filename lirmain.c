/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc.
 * References:
 * The XOR Problem: see LIR p 10
 * The Encoding Problem: see LIR p 14 */

#include <time.h>
#include <stdlib.h>
#include <libc.h>
#include "csv.h"
#include "lir.h"

static double** load(int P, const char* file) {
  Csv* csv = csvnew(file);
  csvload(csv);
  double** pp = malloc(P * sizeof(double*));
  for (int p = 0; p < P; p++) {
    pp[p] = malloc(csv->F * sizeof(double));
    for (int j = 0; j < csv->F; j++) pp[p][j] = atof(csv->r[p][j]);
  }
  csvdel(csv);
  csv = NULL;
  return pp;
}

static void toss(int P, double** pp) {
  for (int p = 0; p < P; p++) free(pp[p]);
  free(pp);
}

static void run(const char* name) {
  // initialize
  char cwd[FLDSIZ];
  getcwd(cwd, sizeof(cwd)); // current working directory
  char buf[FLDSIZ];
  sprintf(buf, "%s/dat/%s.csv", cwd, name); // ~cwd/dat/"name".csv
  Csv* cfgcsv = csvnew(buf);
  csvload(cfgcsv);
  int f = 1; // CSV field; start at 1 to skip network name field
  int C = atoi(cfgcsv->r[1][f++]); // cfgcsv->r[1] holds data, r[0] holds header
  int L = atoi(cfgcsv->r[1][f++]);
  int I = atoi(cfgcsv->r[1][f++]);
  // parse nodes per layer field formatted as "M|N..."
  int N[L * sizeof(int)];
  strcpy(buf, cfgcsv->r[1][f++]);
  int l = 0;
  for (char* t, * s = buf; (t = strtok(s, "|\n\r")) != NULL; s = NULL) N[l++] = atoi(t);
  // parse activation function per layer field formatted as "f|g..."
  strcpy(buf, cfgcsv->r[1][f++]);
  l = 0;
  char* act[L * sizeof(Act)];
  for (char* t, * s = buf; (t = strtok(s, "|\n\r")) != NULL; s = NULL) act[l++] = strndup(t, FLDSIZ); // malloc()
  double eta = atof(cfgcsv->r[1][f++]);
  double alpha = atof(cfgcsv->r[1][f++]);
  double epsilon = atof(cfgcsv->r[1][f++]);
  int P = atoi(cfgcsv->r[1][f++]);
  bool shuffle = istrue(cfgcsv->r[1][f++]);
  csvdel(cfgcsv);
  cfgcsv = NULL;
  // load pattern vectors
  sprintf(buf, "%s/dat/%s-i.csv", cwd, name);
  double** ii = load(P, buf);
  sprintf(buf, "%s/dat/%s-t.csv", cwd, name);
  double** tt = load(P, buf);
  // train network
  Ebp* ebp = ebpnew(name, eta, alpha, epsilon, C, P, shuffle, L, I, N, act);
  learn(ebp, ii, tt);
  dump(ebp);
  recall(ebp, P, ii, tt);
  ebpdel(ebp);
  ebp = NULL;
  // terminate
  toss(P, tt);
  tt = NULL;
  toss(P, ii);
  ii = NULL;
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
