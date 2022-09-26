/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc. */

#ifndef BP_CSV_H
#define BP_CSV_H

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

#endif // BP_CSV_H
