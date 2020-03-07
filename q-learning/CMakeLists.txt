cmake_minimum_required(VERSION 3.10)

project(QLearning VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(q-learning 
    main.cpp
    environments/grid_world.cpp
    q-learning/q-learning.cpp
    )