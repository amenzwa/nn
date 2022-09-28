CC=cc
CFLAGS=-std=c2x -O3

csv.o:	csv.c csv.h
	${CC} ${CFLAGS} -c csv.c

etc.o:	etc.c etc.h
	${CC} ${CFLAGS} -c etc.c

#vec.o:	vec.c vec.h
#	${CC} ${CFLAGS} -c vec.c

lir.o:	lir.c lir.h
	${CC} ${CFLAGS} -c lir.c

lirmain.o:	lirmain.c
	${CC} ${CFLAGS} -c lirmain.c

lir:	lirmain.o lir.o etc.o csv.o
	${CC} ${CFLAGS} lirmain.o lir.o etc.o csv.o -o lir

#ans:	ansmain.o ans.o vec.o etc.o csv.o
#	${CC} ${CFLAGS} ansmain.o ans.o vec.o etc.o csv.o -o ans

#som:	sommain.o som.o vec.o etc.o csv.o
#	${CC} ${CFLAGS} sommain.o som.o vec.o etc.o csv.o -o som

all:	lir	# ans som

clean:
	rm -f *.o lir # ans som