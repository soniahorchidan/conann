#!/bin/sh

# ./build/eval/bound gist 500 500 100 0.1

# ./build/eval/error bert 500 500 100 0.4 > "../Faiss_efficiency_bert_100_0.4.log"

# ./build/eval/error bert_10 0.8 0.6

./build/eval/latency bert_10 0.5 0.1

# ./build/eval/bound bert 500 500 100 0.1

# ./build/eval/bound bert 500 500 100 0.2

# ./build/eval/bound bert 500 500 100 0.3

# ./build/eval/bound bert 500 500 100 0.4

# ./build/eval/bound sift1M 5000 5000 100 0.1