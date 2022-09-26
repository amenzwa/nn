/* Author: Amen Zwa, Esq.
 * Copyright 2022 sOnit, Inc.
 * See section 4.1 Comma-Separated Values,
 * The Practice of Programming, Kernighan (1999) */

#include <stdlib.h>
#include <string.h>
#include "csv.h"

Csv* newCsv(const char* name) {
  Csv* csv = malloc(1 * sizeof(Csv));
  csv->name = strdup(name); // malloc()
  csv->R = csv->F = 0;
  csv->r = NULL;
  return csv;
}

void delCsv(Csv* csv) {
  if (csv->r != NULL) {
    for (int r = 0; r < csv->R; r++) {
      for (int f = 0; f < csv->F; f++) free(csv->r[r][f]);
      free(csv->r[r]);
    }
    free(csv->r);
  }
  free(csv->name);
  free(csv);
}

static char* unquote(char* t) {
  if (t[0] == '"') {
    size_t end = strlen(t) - 1;
    if (t[end] == '"') t[end] = '\0';
    t++;
  }
  return t;
}

void loadCsv(Csv* csv) {
  FILE* fi = fopen(csv->name, "r");
  if (fi == NULL) {
    fprintf(stderr, "ERROR: cannot load CSV file %s\n", csv->name);
    exit(1);
  }
  char rec[RECLEN];
  // count records
  csv->R = 0;
  csv->F = 0;
  while (fgets(rec, sizeof(rec), fi) != NULL) {
    csv->R++;
    if (csv->F == 0) for (char* s = rec; strtok(s, ",\n\r") != NULL; s = NULL) csv->F++;
  }
  rewind(fi);
  // load records
  csv->r = malloc(csv->R * sizeof(char*));
  for (int r = 0; r < csv->R; r++) {
    if (fgets(rec, sizeof(rec), fi) == NULL) {
      fprintf(stderr, "ERROR: cannot load all the records from CSV file %s\n", csv->name);
      exit(1);
    }
    csv->r[r] = malloc(csv->F * sizeof(char*));
    int f = 0;
    for (char* t, * s = rec; (t = strtok(s, ",\n\r")) != NULL; s = NULL) csv->r[r][f++] = strdup(unquote(t)); // malloc()
  }
  fclose(fi);
}

void saveCsv(Csv* csv) {
  FILE* fo = fopen(csv->name, "w");
  if (fo == NULL) {
    fprintf(stderr, "ERROR: cannot save CSV file %s\n", csv->name);
    exit(1);
  }
  for (int r = 0; r < csv->R; r++)
    for (int f = 0; f < csv->F; f++) fprintf(fo, "%s%s", f == 0 ? "" : ",", csv->r[r][f]);
  fclose(fo);
}