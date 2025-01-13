add_executable(bound EXCLUDE_FROM_ALL bound.cpp)
target_link_libraries(bound PRIVATE faiss)

add_executable(demo EXCLUDE_FROM_ALL demo.cpp)
target_link_libraries(demo PRIVATE faiss)

add_executable(effect_error EXCLUDE_FROM_ALL effect_error.cpp)
target_link_libraries(effect_error PRIVATE faiss)