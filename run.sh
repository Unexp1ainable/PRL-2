#!/bin/bash

#pocet cisel bud zadam nebo 10 :)
if [ $# -lt 1 ];then 
    numbers=4;
else
    numbers=$1;
fi;

if [ $numbers -gt 42 ];then 
    numbers=42;
fi;

if [ $numbers -lt 4 ];then 
    numbers=4;
fi;

#preklad cpp zdrojaku
mpic++ --prefix /usr/local/share/OpenMPI -o parkmeans parkmeans.cpp


#vyrobeni souboru s random cisly
dd if=/dev/random bs=1 count=$numbers of=numbers

#spusteni
mpirun --oversubscribe --use-hwthread-cpus --prefix /usr/local/share/OpenMPI -np $numbers parkmeans

#uklid
rm -f oems numbers
