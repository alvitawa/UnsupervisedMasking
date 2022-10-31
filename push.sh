#!/bin/bash
if [ "$#" -eq  "0" ]
  then
    echo "No arguments supplied"
else
    rsync --links -r --exclude-from=.rsyncignore -e ssh --delete . lisa:$2
fi


