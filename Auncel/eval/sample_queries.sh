#!/bin/sh

./../build/eval/sample_queries ../../data/gist/gist_base.fvecs 0.15

./../build/eval/sample_queries ../../data/glove/db.fvecs 0.15

./../build/eval/sample_queries ../../data/deep/deep10M.fvecs 0.15