cmake_minimum_required(VERSION 3.16)
project(HelloWorld)

enable_language(OBJCXX)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

add_executable(helloworld helloworld.m)
target_link_libraries(helloworld ${FOUNDATION_LIBRARY})

add_executable(fib fib.m)
target_link_libraries(fib ${FOUNDATION_LIBRARY})
