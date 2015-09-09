#!/bin/bash
version=3.3.4
FFTWPATH=~/software
mkdir -p ${FFTWPATH}/src
cd ${FFTWPATH}/src

wget http://www.fftw.org/fftw-${version}.tar.gz
tar xvfz fftw-${version}.tar.gz
cd fftw-${version}

./configure --enable-sse2 --enable-avx --enable-fma --enable-shared --enable-threads --prefix=${FFTWPATH} --enable-single && make && make install
./configure --enable-sse2 --enable-avx --enable-fma --enable-shared --enable-threads --prefix=${FFTWPATH} && make && make install
./configure --enable-shared --enable-threads --prefix=${FFTWPATH} --enable-long-double && make && make install
