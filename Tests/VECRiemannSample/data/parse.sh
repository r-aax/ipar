#!/bin/sh

FILE="rrreeesss2.txt"

cat $FILE | awk '{ printf("%f,", $1) } END { printf("\n") }' > dl.txt
cat $FILE | awk '{ printf("%f,", $2) } END { printf("\n") }' > ul.txt
cat $FILE | awk '{ printf("%f,", $3) } END { printf("\n") }' > pl.txt
cat $FILE | awk '{ printf("%f,", $4) } END { printf("\n") }' > cl.txt
cat $FILE | awk '{ printf("%f,", $5) } END { printf("\n") }' > dr.txt
cat $FILE | awk '{ printf("%f,", $6) } END { printf("\n") }' > ur.txt
cat $FILE | awk '{ printf("%f,", $7) } END { printf("\n") }' > pr.txt
cat $FILE | awk '{ printf("%f,", $8) } END { printf("\n") }' > cr.txt
cat $FILE | awk '{ printf("%f,", $9) } END { printf("\n") }' > pm.txt
cat $FILE | awk '{ printf("%f,", $10) } END { printf("\n") }' > um.txt
cat $FILE | awk '{ printf("%f,", $11) } END { printf("\n") }' > d.txt
cat $FILE | awk '{ printf("%f,", $12) } END { printf("\n") }' > u.txt
cat $FILE | awk '{ printf("%f,", $13) } END { printf("\n") }' > p.txt
