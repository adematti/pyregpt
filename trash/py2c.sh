#!/bin/bash

filename=$1
for variable in 'r' 'x'
do
	for i in `seq 2 20`
	do
		echo 's/'$variable'\*\*'$i'/'$variable'['$i']/g'
		sed -i 's/'$variable'\*\*'$i'/'$variable'['$i']/g' $filename
	done
	#sed -i 's/'$variable'\([^[123].*\|$\)/'$variable'[1]\1/g' $filename
done

