#!/bin/bash

makefiles=`find . -name Makefile`
for file in $makefiles; do 
  make -C `dirname $file` clean
done
