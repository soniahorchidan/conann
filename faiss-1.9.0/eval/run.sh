#!/bin/sh

# Error validity + efficiency
# ./build/eval/error bert_100 0.5 0.2 0.1 0.05
# ./build/eval/error glove_100 0.5 0.2 0.1 0.05
# ./build/eval/error gist 0.5 0.2 0.1 0.05
# ./build/eval/error deep10M 0.5 0.2 0.1 0.05
# ./build/eval/error sift10M 0.5 0.2 0.1 0.05

# Variable k
# ./build/eval/error bert_10 0.5 0.2 0.1
# ./build/eval/error bert_100 0.5 0.2 0.1
# ./build/eval/error bert_1000 0.5 0.2 0.1

# ./build/eval/error glove_10 0.5 0.1
# ./build/eval/error glove_1000 0.5 0.1

./build/eval/error gist_10 0.5 0.1
# ./build/eval/error gist_100 0.5 0.1
# ./build/eval/error gist_1000 0.5 0.1

# Latency
# ./build/eval/latency bert_100 0.2 0.1 0.05
# ./build/eval/latency glove_100 0.2 0.1 0.05
# ./build/eval/latency deep10M 0.2 0.1 0.05
