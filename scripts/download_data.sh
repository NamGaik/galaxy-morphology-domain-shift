#!/usr/bin/env bash
# Download the two Galaxy10 data sets into data/ (run from anywhere).
set -e
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {
    # $1 = output file, $2 = url; skip if the file is already there
    if [ -s "data/$1" ]; then
        echo "data/$1 already exists, skipping"
    else
        echo "downloading $1"
        curl -L --fail -o "data/$1" "$2"
    fi
}

fetch Galaxy10_DECals.h5 "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"
fetch Galaxy10.h5 "https://www.astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5"
ls -lh data/
