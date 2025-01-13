# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(bound EXCLUDE_FROM_ALL bound.cpp)
target_link_libraries(bound PRIVATE faiss)

add_executable(demo EXCLUDE_FROM_ALL demo.cpp)
target_link_libraries(demo PRIVATE faiss)