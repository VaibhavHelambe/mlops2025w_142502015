#!/bin/bash

# Check if filename is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 filename"
    exit 1
fi

FILE=$1

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "File not found!"
    exit 1
fi

# Count lines, words, characters
lines=$(wc -l < "$FILE")
words=$(wc -w < "$FILE")
chars=$(wc -c < "$FILE")

echo "File: $FILE"
echo "Number of lines: $lines"
echo "Number of words: $words"
echo "Number of characters: $chars"
