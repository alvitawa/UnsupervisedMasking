echo "Job STARTED at `date`"

cp -Lr "$HOME/workspace/data/$1" "$TMPDIR"

ls -l "$TMPDIR"

# The first argument was the dataset to move
# The remaining arguments are the command to run

# Run the command, ignoring the first argument
shift 1
# Run the command, but append a parameter pointing to the new location of the dataset
"$@" "main.data_path=$TMPDIR"
