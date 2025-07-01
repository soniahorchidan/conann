add_executable(latency EXCLUDE_FROM_ALL latency.cpp)
target_link_libraries(latency PRIVATE faiss)

add_executable(error EXCLUDE_FROM_ALL error.cpp)
target_link_libraries(error PRIVATE faiss)

add_executable(error_pq EXCLUDE_FROM_ALL error_pq.cpp)
target_link_libraries(error_pq PRIVATE faiss)

add_executable(test_cache EXCLUDE_FROM_ALL test_cache.cpp)
target_link_libraries(test_cache PRIVATE faiss)

add_executable(variable_k EXCLUDE_FROM_ALL variable_k.cpp)
target_link_libraries(variable_k PRIVATE faiss)