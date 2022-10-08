/* Author: Amen Zwa, Esq.
 * Copyright (c) 2022 sOnit, Inc.
 * See section 4.1 Comma-Separated Values,
 * The Practice of Programming, Kernighan (1999) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "csv.h"

Csv* csvnew(const char* name) {
  Csv* csv = malloc(sizeof(Csv));
  csv->name = strndup(name, FLDSIZ); // malloc()
  csv->R = csv->F = 0;
  csv->r = NULL;
  return csv;
}

void csvdel(Csv* csv) {
  if (csv->r != NULL) {
    for (int r = 0; r < csv->R; r++) {
      for (int f = 0; f < csv->F; f++) free(csv->r[r][f]);
      free(csv->r[r]);
    }
    free(csv->r);
    csv->r = NULL;
  }
  free(csv->name);
  csv->name = NULL;
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

void csvload(Csv* csv) {
  FILE* fi = fopen(csv->name, "r");
  if (fi == NULL) {
    fprintf(stderr, "ERROR: cannot load CSV file %s\n", csv->name);
    exit(1);
  }
  // count records
  char rec[RECSIZ];
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
    for (char* t, * s = rec; (t = strtok(s, ",\n\r")) != NULL; s = NULL) csv->r[r][f++] = strndup(unquote(t), FLDSIZ); // malloc()
  }
  fclose(fi);
}

void csvsave(Csv* csv) {
  FILE* fo = fopen(csv->name, "w");
  if (fo == NULL) {
    fprintf(stderr, "ERROR: cannot save CSV file %s\n", csv->name);
    exit(1);
  }
  for (int r = 0; r < csv->R; r++)
    for (int f = 0; f < csv->F; f++) fprintf(fo, "%s%s", f == 0 ? "" : ",", csv->r[r][f]);
  fclose(fo);
}