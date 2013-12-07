#!/bin/bash

N=128
if [ $1 ]; then
	N=$1
fi
TBLOQUE=64
if [ $2 ]; then
	TBLOQUE=$2
fi

if [ -e optimizado-shared ]; then
	echo ">>> OPTIMIZADO C/ SHARED CON N=$N y TBLOQUE=$TBLOQUE"
	./optimizado-shared $N $TBLOQUE
	printf "\n"
fi

if [ -e optimizado ]; then
	echo ">>> OPTIMIZADO CON N=$N y TBLOQUE=$TBLOQUE"
	./optimizado $N $TBLOQUE
	printf "\n"
fi

if [ -e no-optimizado ]; then
	echo ">>> NO OPTIMIZADO CON N=$N y TBLOQUE=$TBLOQUE"
	./no-optimizado $N $TBLOQUE
	printf "\n"
fi

if [ -e secuencial ]; then
	echo ">>> SECUENCIAL CON N=$N"
	./secuencial $N
	printf "\n"
fi
