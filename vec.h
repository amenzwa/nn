/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#ifndef NN_VEC_H
#define NN_VEC_H

typedef struct Vec {
  int C; // number of components
  double* c; // components
} Vec;

typedef struct Mat {
  int R, C; // number of rows and columns
  Vec** r; // row vectors
} Mat;

extern Vec* vecnew(int C);
extern void vecdel(Vec* v);
extern void veccpy(Vec* o, const Vec* v);
extern void vecadd(Vec* o, const Vec* u, const Vec* v);
extern void vecsub(Vec* o, const Vec* u, const Vec* v);
extern void vecscale(Vec* o, double s, const Vec* v);
extern void vecouter(Mat* o, const Vec* u, const Vec* v);
extern double vecinner(const Vec* u, const Vec* v);
extern double veceuclidean(const Vec* u, const Vec* v);
extern void vecmap(Vec* o, double (* f)(double), int C, const Vec* v);
extern double vecfold(double (* f)(double, double), double unit, const Vec* v);
extern void veczipwith(Vec* o, double (* f)(double, double), const Vec* u, const Vec* v);
extern Mat* matnew(int R, int C);
extern void matdel(Mat* m);
extern void mattr(Mat* o, const Mat* m);
extern void matcol(Vec* o, int c, const Mat* m);
extern void matadd(Mat* o, const Mat* m, const Mat* n);
extern void matmul(Vec* o, const Mat* m, const Vec* v);
extern void matscale(Mat* o, double s, const Mat* m);

#endif // NN_VEC_H
