#!/bin/bash

N=128
if [ $1 ]; then
	N=$1
fi
TBLOQUE=64
if [ $2 ]; then
	TBLOQUE=$2
fi
COUNT=1
if [ $3 ]; then
	COUNT=$3
fi

if [ -e optimizado-shared ]; then
	echo ">>> OPTIMIZADO C/ SHARED (X$COUNT) CON N=$N y TBLOQUE=$TBLOQUE"
	./optimizado-shared $N $TBLOQUE $COUNT
	printf "\n"
fi

if [ -e optimizado ]; then
	echo ">>> OPTIMIZADO (X$COUNT) CON N=$N y TBLOQUE=$TBLOQUE"
	./optimizado $N $TBLOQUE $COUNT
	printf "\n"
fi

if [ -e no-optimizado ]; then
	echo ">>> NO OPTIMIZADO (X$COUNT) CON N=$N y TBLOQUE=$TBLOQUE"
	./no-optimizado $N $TBLOQUE $COUNT
	printf "\n"
fi

if [ -e secuencial ]; then
	echo ">>> SECUENCIAL (X$COUNT) CON N=$N"
	./secuencial $N $COUNT
	printf "\n"
fi
