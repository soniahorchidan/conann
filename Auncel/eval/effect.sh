#!/bin/sh

# ./build/eval/effect_error bert 100 0.5 0.5 0.3 0.2 0.1 0.05
# ./build/eval/effect_error bert 10 0.5 0.5 0.3 0.2 0.1 0.05
# ./build/eval/effect_error bert 1000 0.5 0.5 0.3 0.2 0.1 0.05

./build/eval/effect_error gist 100 0.5 0.5 0.3 0.2 0.1 0.05
./build/eval/effect_error gist 10 0.5 0.5 0.3 0.2 0.1 0.05
./build/eval/effect_error gist 1000 0.5 0.5 0.3 0.2 0.1 0.05

./build/eval/effect_error sift1M 100 0.5 0.5 0.3 0.2 0.1 0.05
./build/eval/effect_error sift1M 10 0.5 0.5 0.3 0.2 0.1 0.05
./build/eval/effect_error sift1M 1000 0.5 0.5 0.3 0.2 0.1 0.05

./build/eval/effect_error deep10M 100 0.5 0.5 0.3 0.2 0.1 0.05
./build/eval/effect_error deep10M 10 0.5 0.5 0.3 0.2 0.1 0.05
./build/eval/effect_error deep10M 1000 0.5 0.5 0.3 0.2 0.1 0.05

