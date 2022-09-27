CC=cc
CFLAGS=-O3

csv.o: csv.c csv.h
	${CC} ${CFLAGS} -c csv.c

vec.o: vec.c vec.h
	${CC} ${CFLAGS} -c vec.c

lir.o: lir.c lir.h
	${CC} ${CFLAGS} -c lir.c

lirmain.o:	lirmain.c
	${CC} ${CFLAGS} -c lirmain.c

lir:	lirmain.o lir.o csv.o
	${CC} ${CFLAGS} lirmain.o lir.o csv.o -o lir

ans:	ansmain.o ans.o vec.o csv.o
	${CC} ${CFLAGS} ansmain.o ans.o vec.o csv.o -o ans

all:	lir	ans

clean:
	rm -f *.o lir ans