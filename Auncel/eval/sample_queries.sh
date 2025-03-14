#!/bin/sh

# ./../build/eval/sample_queries ../../data/bert/db.fvecs 0.15

./../build/eval/sample_queries ../../data/gist/gist_base.fvecs 30000 gist_small_sample.fvecs
./../build/eval/sample_queries ../../data/gist/gist_base.fvecs 5000 gist_small_sample_queries.fvecs

./../build/eval/sample_queries ../../data/glove/db.fvecs 30000 glove_small_sample.fvecs
./../build/eval/sample_queries ../../data/glove/db.fvecs 5000 glove_small_sample_queries.fvecs

# ./../build/eval/sample_queries ../../data/deep/deep10M.fvecs 0.15

# ./../build/eval/sample_queries ../../data/next-sift/sift_base.fvecs 0.15