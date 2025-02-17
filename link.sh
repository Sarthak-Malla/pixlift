#!/bin/bash
# Run this script after running setup.sh

thorium_src="$HOME/thorium/src"
chromium_src="$HOME/chromium/src"

# Get the files in thorium_src which were modified or added after last git commit
modified_files=$(git -C "$thorium_src" ls-files --modified --others)

for file in $modified_files; do
    echo "$file"

    thorium_file="$thorium_src/$file"

    # Check if the file exists in chromium_src
    chromium_file="$chromium_src/$file"

    if [ -f "$chromium_file" ]; then
        # Compare md5 hashes
        thorium_hash=$(md5sum "$thorium_file" | awk '{print $1}')
        chromium_hash=$(md5sum "$chromium_file" | awk '{print $1}')

        delete_chromium_file=true

        if [ "$thorium_hash" != "$chromium_hash" ]; then
            read -p "  Hashes do not match. Overwrite file in chromium? (y/n)" choice
            case $choice in
                [yY]) delete_chromium_file=true;;
                [nN]) delete_chromium_file=false;;
                *) echo "  Invalid response; skipping"; delete_chromium_file=false;;
            esac
        fi

        if $delete_chromium_file; then
            # Delete the second file in preparation for creating hard link
            echo -n "  "
            rm -v "$chromium_file"
        else
            continue
        fi
    else
        # File doesn't exist. Create parent directories if they don't exist
        dir_chromium_file=$(dirname "$chromium_file")
        if [ ! -f "$dir_chromium_file" ]; then
            echo "  Creating parent directories: $dir_chromium_file"
            mkdir -p "$dir_chromium_file"
        fi
    fi

    # Create hard link
    echo -n "  ";
    ln -v "$thorium_file" "$chromium_file"
done

