cmake_minimum_required(VERSION 3.10)

project(QLearning VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(q-learning 
    main.cpp 
    q-learning/q-learning.cpp
    environments/grid_world.cpp
    vis/q-learning_vis.cpp)

target_link_libraries (q-learning sfml-graphics sfml-window sfml-system)