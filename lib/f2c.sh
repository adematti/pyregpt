#!/bin/bash

filename=$1
for variable in 'k1' 'k2' 'k3' 'q' 'k'
do
	for i in `seq 2 20`
	do
		echo 's/'$variable'\*\*'$i'/'$variable'['$i']/g'
		sed -i 's/'$variable'\*\*'$i'/'$variable'['$i']/g' $filename
	done
	#sed -i 's/'$variable'\([^[123].*\|$\)/'$variable'[1]\1/g' $filename
done
sed -i 's/.d0/./g' $filename
sed -i 's/pi/M_PI/g' $filename
sed -i 's/d6/e6/g' $filename
sed -i 's/dlog/my_log/g' $filename
sed -i 's/dsqrt/my_sqrt/g' $filename
sed -i 's/dabs/my_abs/g' $filename

