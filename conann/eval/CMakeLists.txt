add_executable(latency EXCLUDE_FROM_ALL latency.cpp)
target_link_libraries(latency PRIVATE faiss)

add_executable(error EXCLUDE_FROM_ALL error.cpp)
target_link_libraries(error PRIVATE faiss)