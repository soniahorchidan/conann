#!/bin/sh

# ./../build/eval/effect_error sift10k 10 0.66 0.05
# ./../build/eval/effect_error sift10k 10 0.66 0.1
# ./../build/eval/effect_error sift10k 10 0.66 0.8

# ./../build/eval/effect_error sift1M 100 5000 5000

./../build/eval/effect_error bert100 100 0.5 0.9
./../build/eval/effect_error bert100 100 0.5 0.2

# ./../build/eval/effect_error deep10M 100 5000 5000

# ./../build/eval/effect_error gist100 100 5000 5000

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