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

Vec* vecnew(int C) {
  /* Create a [C] vector. */
  Vec* v = malloc(1 * sizeof(Vec));
  v->C = C;
  v->c = calloc(v->C, sizeof(double));
  return v;
}

void vecdel(Vec* v) {
  /* Destroy the vector. */
  free(v->c);
  free(v);
}

inline void veccpy(const Vec* v, Vec* o) {
  /* [o] = [v] (copy only v->C components) */
  memcpy(o->c, v->c, v->C * sizeof(double));
}

inline void vecadd(const Vec* u, const Vec* v, Vec* o) {
  /* [o] = [u] + [v] */
  for (int c = 0; c < u->C; c++) o->c[c] = u->c[c] + v->c[c];
}

inline void vecsub(const Vec* u, const Vec* v, Vec* o) {
  /* [o] = [u] - [v] */
  for (int c = 0; c < u->C; c++) o->c[c] = u->c[c] - v->c[c];
}

inline void vecscale(double s, const Vec* v, Vec* o) {
  /* [o] = s * [v] */
  for (int c = 0; c < v->C; c++) o->c[c] = s * v->c[c];
}

void vecouter(const Vec* u, const Vec* v, Mat* o) {
  /* (o) = [u]' * [v] */
  for (int r = 0; r < u->C; r++)
    for (int c = 0; c < v->C; c++) o->r[r]->c[c] = u->c[r] * v->c[c];
}

double vecinner(const Vec* u, const Vec* v) {
  /* dot = [u] . [v] */
  double dot = 0.0;
  for (int c = 0; c < u->C; c++) dot += u->c[c] * v->c[c];
  return dot;
}

double veceuclidean(const Vec* u, const Vec* v) {
  /* euc = ||[u] - [v]|| */
  double euc = 0.0;
  for (int c = 0; c < u->C; c++) euc += sqre(u->c[c] - v->c[c]);
  return sqrt(euc);
}

inline void vecmap(double (*f)(double), int C, const Vec* v, Vec* o) {
  /* [o] = f [v] */
  for (int c = 0; c < C; c++) o->c[c] = f(v->c[c]);
}

double vecfold(double (*f)(double, double), double unit, const Vec* v) {
  /* acc = foldl f unit v */
  double acc = unit; // unit of monoid
  for (int c = 0; c < v->C; c++) acc = f(acc, v->c[c]);
  return acc;
}

inline void veczipwith(double (*f)(double, double), const Vec* u, const Vec* v, Vec* o) {
  /* [o] = [u] `f` [v] */
  for (int c = 0; c < u->C; c++) o->c[c] = f(u->c[c], v->c[c]);
}

/* matrix */

Mat* matnew(int R, int C) {
  /* Create an (R x C) matrix. */
  Mat* m = malloc(1 * sizeof(Mat));
  m->R = R;
  m->C = C;
  m->r = malloc(m->R * sizeof(Vec*));
  for (int r = 0; r < m->R; r++) m->r[r] = vecnew(C);
  return m;
}

void matdel(Mat* m) {
  /* Destroy the matrix. */
  for (int r = 0; r < m->R; r++) vecdel(m->r[r]);
  free(m->r);
  free(m);
}

void mattr(const Mat* m, Mat* o) {
  /* (o) = (m)' */
  for (int r = 0; r < m->R; r++)
    for (int c = 0; c < m->C; c++) o->r[c]->c[r] = m->r[r]->c[c];
}

inline void matcol(int c, const Mat* m, Vec* o) {
  /* [o] = m(*, c) */
  for (int r = 0; r < m->R; r++) o->c[r] = m->r[r]->c[c];
}

inline void matadd(const Mat* m, const Mat* n, Mat* o) {
  /* (o) = (m) + (n) */
  for (int r = 0; r < m->R; r++) vecadd(m->r[r], n->r[r], o->r[r]);
}

inline void matmul(const Mat* m, const Vec* v, Vec* o) {
  /* [o] = (m) * [v] */
  for (int r = 0; r < v->C; r++) o->c[r] = vecinner(m->r[r], v);
}

inline void matscale(double s, const Mat* m, Mat* o) {
  /* (o) = s * (m) */
  for (int r = 0; r < m->R; r++) vecscale(s, m->r[r], o->r[r]);
}