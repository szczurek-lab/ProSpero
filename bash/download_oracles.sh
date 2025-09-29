#!/bin/bash

FILE_ID="1UiKVdnNDlqeHMHvc12qd6kbw5QltQm9D"
OUT=out.zip
gdown $FILE_ID -O $OUT
unzip $OUT -d .
rm $OUT

