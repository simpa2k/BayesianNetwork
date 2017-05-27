cmake_minimum_required(VERSION 3.7)
project(graph)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp directedGraph/Graph.h tests/catch.h tests/graphTest.cpp bayesNet/BayesianNetwork.cpp bayesNet/BayesianNetwork.h bayesNet/brain/Brain.cpp bayesNet/brain/Brain.h bayesNet/utilities/utilities.cpp bayesNet/utilities/utilities.h tests/bayesianNetworkTest.cpp)
add_executable(graph ${SOURCE_FILES})
target_link_libraries(graph ${ARMADILLO_LIBRARIES})