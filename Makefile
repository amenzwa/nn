# Author: Amen Zwa, Esq.
# Copyright (c) 2022 sOnit, Inc.

CC=cc
CFLAGS=-std=c2x -O3 # -g

# utilities

csv.o:	csv.c csv.h
	${CC} ${CFLAGS} -c csv.c

vec.o:	vec.c vec.h
	${CC} ${CFLAGS} -c vec.c

etc.o:	etc.c etc.h
	${CC} ${CFLAGS} -c etc.c

# LIR

lir.o:	lir.c lir.h etc.h csv.h
	${CC} ${CFLAGS} -c lir.c

lirmain.o:	lirmain.c lir.h csv.h
	${CC} ${CFLAGS} -c lirmain.c

lir:	lirmain.o lir.o etc.o csv.o
	${CC} ${CFLAGS} lirmain.o lir.o etc.o csv.o -o lir

# SOM

som.o:	som.c som.h vec.h
	${CC} ${CFLAGS} -c som.c

sommain.o:	sommain.c som.h etc.h csv.h
	${CC} ${CFLAGS} -c sommain.c

som:	sommain.o som.o vec.o etc.o csv.o
	${CC} ${CFLAGS} sommain.o som.o vec.o etc.o csv.o -o som

# miscellaneous

all:	lir som

clean:
	rm -f *.o lir som