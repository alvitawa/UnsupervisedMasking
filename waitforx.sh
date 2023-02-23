#!/bin/sh
PID="$1"
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running" 
    sleep 2
done
echo "Process $PID has finished" 

