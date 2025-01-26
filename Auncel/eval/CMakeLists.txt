add_executable(bound EXCLUDE_FROM_ALL bound.cpp)
target_link_libraries(bound PRIVATE faiss)

add_executable(demo EXCLUDE_FROM_ALL demo.cpp)
target_link_libraries(demo PRIVATE faiss)

add_executable(effect_error EXCLUDE_FROM_ALL effect_error.cpp)
target_link_libraries(effect_error PRIVATE faiss)

add_executable(effect_time EXCLUDE_FROM_ALL effect_time.cpp)
target_link_libraries(effect_time PRIVATE faiss)

add_executable(overhead EXCLUDE_FROM_ALL overhead.cpp)
target_link_libraries(overhead PRIVATE faiss)

add_executable(latency EXCLUDE_FROM_ALL latency.cpp)
target_link_libraries(latency PRIVATE faiss)

add_executable(compute_gt EXCLUDE_FROM_ALL compute_gt.cpp)
target_link_libraries(compute_gt PRIVATE faiss)

add_executable(sample_queries EXCLUDE_FROM_ALL sample_queries.cpp)
target_link_libraries(sample_queries PRIVATE faiss)