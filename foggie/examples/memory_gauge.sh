#!/bin/bash
count=1
echo "---------------------------Nodes------------------------"
/u/scicon/tools/bin/qsh.pl $1 "cat /proc/meminfo"   | grep r  | grep i | grep n | grep -v SU | awk '{print $1}' | fmt -1000
echo "-----------------------Total Memory---------------------"
/u/scicon/tools/bin/qsh.pl $1 "cat /proc/meminfo" | grep MemTotal: | awk '{print $2/1024/1024}' | fmt -1000
echo "----------------------Available Memory---------------------"
while [ $count -le 99999 ]
do
    sleep 10
       /u/scicon/tools/bin/qsh.pl $1 "cat /proc/meminfo" | grep Available: | awk '{print $2/1024/1024}' | fmt -1000
    ((count++))
done
