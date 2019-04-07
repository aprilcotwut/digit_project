#!/bin/bash
ls ../ | grep txt > files.in
python3 conv.py files.in dat.csv
rm files.in
cp dat.csv ../plot/dat.csv
