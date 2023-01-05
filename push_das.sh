#!/bin/bash
rsync --links -r --exclude-from=.rsyncignore -e ssh --delete . das:workspace
