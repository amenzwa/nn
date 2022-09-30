/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#ifndef NN_VEC_H
#define NN_VEC_H

typedef struct Vec {
  int C; // number of components
  double* c; // components
} Vec;

extern Vec* newVec(int C);
extern void delVec(Vec* v);
extern void cpyVec(const Vec* v, Vec* o);
extern void addVVV(const Vec* u, const Vec* v, Vec* o);
extern void subVVV(const Vec* u, const Vec* v, Vec* o);
extern void mulSVV(double s, const Vec* v, Vec* o);
extern void mulVVV(const Vec* u, const Vec* v, Vec* o);
extern double dotVVS(const Vec* u, const Vec* v);
extern double eucVVS(const Vec* u, const Vec* v);
extern void mapVec(double (*f)(double), int C, const Vec* v, Vec* o);
extern double foldVec(double (*f)(double, double), double unit, const Vec* v);

typedef struct Mat {
  int R, C; // number of rows and columns
  Vec** r; // row vectors
} Mat;

extern Mat* newMat(int R, int C);
extern void delMat(Mat* m);
extern void trnMat(const Mat* m, Mat* o);
extern void colVec(int c, const Mat* m, Vec* o);
extern void addMMM(const Mat* m, const Mat* n, Mat* o);
extern void mulSMM(double s, const Mat* m, Mat* o);
extern void mulVVM(const Vec* u, const Vec* v, Mat* o);
extern void mulMVV(const Mat* m, const Vec* v, Vec* o);

#endif // NN_VEC_H
