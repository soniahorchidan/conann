#!/bin/sh

# ./../build/eval/effect_error sift10k 10 0.66 
# ./../build/eval/effect_error sift10k 10 0.66
# ./../build/eval/effect_error sift10k 10 0.66

./../build/eval/effect_error bert_100 100 0.66 0.2
./../build/eval/effect_error bert_100 100 0.66 0.1
./../build/eval/effect_error bert_100 100 0.66 0.05

./../build/eval/effect_error glove_100 100 0.66 0.2
./../build/eval/effect_error glove_100 100 0.66 0.1
./../build/eval/effect_error glove_100 100 0.66 0.05

./../build/eval/effect_error gist_100 100 0.66 0.2
./../build/eval/effect_error gist_100 100 0.66 0.1
./../build/eval/effect_error gist_100 100 0.66 0.05

# variable k
./../build/eval/effect_error bert_10 10 0.66 0.1
./../build/eval/effect_error bert_1000 1000 0.66 0.1

./../build/eval/effect_error glove_10 10 0.66 0.1
./../build/eval/effect_error glove_1000 1000 0.66 0.1

./../build/eval/effect_error gist_10 10 0.66 0.1
./../build/eval/effect_error gist_1000 1000 0.66 0.1

# BEGIN AUNCEL BLOCK
# ./effect_error sift10M 100 5000 5000

# ./effect_error deep10M 100 5000 5000

# ./effect_error gist 100 500 500

# ./effect_error text 100 5000 5000

# ./effect_time sift10M 100 5000 5000

# ./effect_time deep10M 100 5000 5000

# ./effect_time gist 100 500 500

# ./effect_time text 100 5000 5000
# END AUNCEL BLOCK