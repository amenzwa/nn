/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc.
 * References:
 * See chapter 8 Matrices and Vector Spaces,
 * MPE: Mathematical Methods for Physics and Engineering, Riley (2018) */

#include <stdlib.h>
#include <libc.h>
#include <math.h>
#include "etc.h"
#include "vec.h"

/* vector */

Vec* newVec(int C) {
  /* Create a [C] vector. */
  Vec* v = malloc(1 * sizeof(Vec));
  v->C = C;
  v->c = calloc(v->C, sizeof(double));
  return v;
}

void delVec(Vec* v) {
  /* Destroy the vector. */
  free(v->c);
  free(v);
}

inline void cpyVec(const Vec* v, Vec* o) {
  /* [o] = [v] (copy only v->C components) */
  memcpy(o->c, v->c, v->C * sizeof(double));
}

inline void addVVV(const Vec* u, const Vec* v, Vec* o) {
  /* [o] = [u] + [v] */
  for (int c = 0; c < u->C; c++) o->c[c] = u->c[c] + v->c[c];
}

inline void subVVV(const Vec* u, const Vec* v, Vec* o) {
  /* [o] = [u] - [v] */
  for (int c = 0; c < u->C; c++) o->c[c] = u->c[c] - v->c[c];
}

inline void mulSVV(double s, const Vec* v, Vec* o) {
  /* [o] = s * [v] */
  for (int c = 0; c < v->C; c++) o->c[c] = s * v->c[c];
}

inline void mulVVV(const Vec* u, const Vec* v, Vec* o) {
  /* [o] = [u] * [v] (multiply component wise) */
  for (int c = 0; c < u->C; c++) o->c[c] = u->c[c] * v->c[c];
}

double dotVVS(const Vec* u, const Vec* v) {
  /* dot = [u] . [v] */
  double dot = 0.0;
  for (int c = 0; c < u->C; c++) dot += u->c[c] * v->c[c];
  return dot;
}

double eucVVS(const Vec* u, const Vec* v) {
  /* euc = |[u] - [v]| */
  double euc = 0.0;
  for (int c = 0; c < u->C; c++) euc += sqre(u->c[c] - v->c[c]);
  return sqrt(euc);
}

inline void mapVec(double (*f)(double), int C, const Vec* v, Vec* o) {
  /* [o] = f [v] */
  for (int c = 0; c < C; c++) o->c[c] = f(v->c[c]);
}

double foldVec(double (*f)(double, double), double unit, const Vec* v) {
  /* acc = foldl f unit v */
  double acc = unit; // unit of monoid
  for (int c = 0; c < v->C; c++) acc = f(acc, v->c[c]);
  return acc;
}

/* matrix */

Mat* newMat(int R, int C) {
  /* Create an (R x C) matrix. */
  Mat* m = malloc(1 * sizeof(Mat));
  m->R = R;
  m->C = C;
  m->r = malloc(m->R * sizeof(Vec*));
  for (int r = 0; r < m->R; r++) m->r[r] = newVec(C);
  return m;
}

void delMat(Mat* m) {
  /* Destroy the matrix. */
  for (int r = 0; r < m->R; r++) delVec(m->r[r]);
  free(m->r);
  free(m);
}

void trnMat(const Mat* m, Mat* o) {
  /* (o) = (m)' */
  for (int r = 0; r < m->R; r++)
    for (int c = 0; c < m->C; c++) o->r[c]->c[r] = m->r[r]->c[c];
}

inline void colVec(int c, const Mat* m, Vec* o) {
  /* [o] = m(*, c) */
  for (int r = 0; r < m->R; r++) o->c[r] = m->r[r]->c[c];
}

inline void addMMM(const Mat* m, const Mat* n, Mat* o) {
  /* (o) = (m) + (n) */
  for (int r = 0; r < m->R; r++) addVVV(m->r[r], n->r[r], o->r[r]);
}

inline void mulSMM(double s, const Mat* m, Mat* o) {
  /* (o) = s * (m) */
  for (int r = 0; r < m->R; r++) mulSVV(s, m->r[r], o->r[r]);
}

void mulVVM(const Vec* u, const Vec* v, Mat* o) {
  /* (o) = [u]' * [v] */
  for (int r = 0; r < u->C; r++)
    for (int c = 0; c < v->C; c++) o->r[r]->c[c] = u->c[r] * v->c[c];
}

inline void mulMVV(const Mat* m, const Vec* v, Vec* o) {
  /* [o] = (m) * [v] */
  for (int r = 0; r < v->C; r++) o->c[r] = dotVVS(m->r[r], v);
}