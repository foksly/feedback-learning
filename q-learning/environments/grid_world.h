#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

enum class Action { Right, Left, Up, Down, Size = 4 };

typedef int State;
typedef int Reward;

class Maze {
   public:
    Maze(int n_rows, int n_cols, State start, State end);

    Maze(int size, State start, State end);

    Maze(int num_rows, int num_cols, std::pair<int, int> start, std::pair<int, int> end);

    Maze(int size, std::pair<int, int> start, std::pair<int, int> end);

    char operator[](State state);

    size_t Size() const;
    
    int NumberOfRows() const;
    
    int NumberOfCols() const;

    std::vector<char> GetValidForStepValues() const;

    Reward GetRewardInState(State state);

    State GetStartState() const;

    State GetEndState() const;

    const char kGrid = 'G';
    const char kStart = 'S';
    const char kEnd = 'E';
    const char kWall = 'W';
    const char kDoor = 'D';
    const char kKey = 'K';

   private:
    std::vector<char> maze_;
    int n_rows_;
    int n_cols_;

    State start_;
    State end_;

    std::unordered_map<char, Reward> value_to_reward;
};

struct Observation {
    State state;
    Reward reward;
    bool is_done;
};

class Environment {
   public:
    explicit Environment(Maze maze);

    Environment();

    State Reset();

    void Render();

    Observation Step(Action action);

    State GetCurrentState() const;

    int NumberOfStates();

    State SampleState();

    int NumberOfActions();

    Action SampleAction();

    Maze maze_;

   private:
    State current_state_;
    std::mt19937 random_generator;

    State ConvertCoordinateToState(std::pair<int, int> coordinate);

    std::pair<int, int> ConvertStateToCoordinate(State state);

    bool IsValidForStep(State state, std::vector<char> valid_values);

    State GetNextState(State state, Action action);

    Reward GetRewardForAction(State state, Action action);
};