/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc. */

#ifndef NN_CSV_H
#define NN_CSV_H

#define RECLEN 16384 // CSV record length (in bytes)

typedef struct Csv {
  char* name; // file name
  int R; // number of records
  int F; // number of fields per record
  char*** r; // records
} Csv;

extern Csv* csvnew(const char* name);
extern void csvdel(Csv* csv);
extern void csvload(Csv* csv);
extern void csvsave(Csv* csv);

#endif // NN_CSV_H
