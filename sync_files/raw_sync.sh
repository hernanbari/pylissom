source sync_files/raw_sync_path
if [[  -z  $nikola_path ]]; then
    echo "Error set variable  nikola_path on file raw_sync_path"
else
    echo Starting...
    date


    find . -not -path "./venv*" -not -path "./resources/caches*" | grep py$ | grep -v npy > sync_files/files
    find . -not -path "./venv*" -not -path "./resources/caches*" | grep yaml$ >> sync_files/files
    find . -not -path "./venv*" -not -path "./resources/caches*" | grep sql$ >> sync_files/files


    rsync -r --partial --files-from=sync_files/files --rsh=ssh . hbari@nikola.dc.uba.ar:$nikola_path
    echo Ended
    date
fi
