source sync_files/raw_sync_path
if [[  -z  $cerebro_path ]]; then
    echo "Error set variable  cerebro_path on file raw_sync_path"
else
    echo Starting...
    date


    find . -not -path "./venv*" -not -path "./resources/caches*" -not -path "./torch*" -not -path "./pytorch" | grep py$ | grep -v npy > sync_files/files
    find . -not -path "./venv*" -not -path "./resources/caches*" -not -path "./torch*" -not -path "./pytorch" | grep yaml$ >> sync_files/files
    find . -not -path "./venv*" -not -path "./resources/caches*" -not -path "./torch*" -not -path "./pytorch" | grep sql$ >> sync_files/files
    find . -not -path "./venv*" -not -path "./resources/caches*" -not -path "./torch*" -not -path "./pytorch" | grep ini$ >> sync_files/files

    rsync -r --partial --files-from=sync_files/files --rsh=ssh . hbari@cerebro.dc.uba.ar:$cerebro_path
    echo Ended
    date
fi
