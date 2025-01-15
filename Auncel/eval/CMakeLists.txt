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