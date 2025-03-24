#!/bin/bash
#nohup bash inference.sh &
ctx=0
epoch=150
bs=8
lr=1e-5
hidden=128
layer=4
mode=test

python main.py --mode=$mode --ctx=$ctx --bs=$bs --epoch=$epoch --lr=$lr --hidden=$hidden --layer=$layer