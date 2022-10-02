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
extern void veccpy(const Vec* v, Vec* o);
extern void vecadd(const Vec* u, const Vec* v, Vec* o);
extern void vecsub(const Vec* u, const Vec* v, Vec* o);
extern void vecscale(double s, const Vec* v, Vec* o);
extern void vecouter(const Vec* u, const Vec* v, Mat* o);
extern double vecinner(const Vec* u, const Vec* v);
extern double veceuclidean(const Vec* u, const Vec* v);
extern void vecmap(double (*f)(double), int C, const Vec* v, Vec* o);
extern double vecfold(double (*f)(double, double), double unit, const Vec* v);
extern void veczipwith(double (*f)(double, double), const Vec* u, const Vec* v, Vec* o);

extern Mat* matnew(int R, int C);
extern void matdel(Mat* m);
extern void mattr(const Mat* m, Mat* o);
extern void matcol(int c, const Mat* m, Vec* o);
extern void matadd(const Mat* m, const Mat* n, Mat* o);
extern void matmul(const Mat* m, const Vec* v, Vec* o);
extern void matscale(double s, const Mat* m, Mat* o);

#endif // NN_VEC_H
