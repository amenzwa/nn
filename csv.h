/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc. */

#ifndef BPFOR_CSV_H
#define BPFOR_CSV_H

#include <stdio.h>

#define RECLEN 16384 // CSV record length (in bytes)
#define MAXFLD 8192 // maximum number of CSV fields per record

typedef struct Csv {
  char* name; // file name
  int R; // number of records
  int F; // number of fields per record
  char*** r; // records
} Csv;

extern Csv* newCsv(const char* name);
extern void delCsv(Csv* csv);
extern void loadCsv(Csv* csv);
extern void saveCsv(Csv* csv);

#endif // BPFOR_CSV_H
