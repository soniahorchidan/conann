#!/bin/sh

# dataset, calibration size, alpha, nlist, k
./build/eval/error bert 0.5 0.1 128 100
# k is optional, can also run with previous dataset identifiers and will read k out of the GT
./build/eval/error bert_100 0.5 0.1 128
