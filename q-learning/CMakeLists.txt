cmake_minimum_required(VERSION 3.10)

project(QLearning VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(environments/)
add_executable(qlearning 
    main.cpp 
    q-learning/q-learning.cpp
    environments/grid_world.cpp)