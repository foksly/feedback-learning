#pragma once

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// enum class Action { Right, Left, Up, Down, Size = 4 };

typedef int State;
typedef int Reward;

class Maze {
   public:
    Maze(int n_rows, int n_cols, State start, State end);

    Maze(int size, State start, State end);

    Maze(int num_rows, int num_cols, std::pair<int, int> start, std::pair<int, int> end);

    Maze(int size, std::pair<int, int> start, std::pair<int, int> end);

    char operator[](State state) const;

    char& operator[](State state);

    size_t Size() const;

    int NumberOfRows() const;

    int NumberOfCols() const;

    std::vector<char> GetValidForStepValues() const;

    Reward GetRewardInState(State state);

    State GetStartState() const;

    State GetEndState() const;

    State ConvertCoordinateToState(std::pair<int, int> coordinate) const;

    std::pair<int, int> ConvertStateToCoordinate(State state) const;

    void ChangeRewardValue(char key, Reward Reward);

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

    std::unordered_map<char, Reward> value2reward;
};

struct Observation {
    State state;
    Reward reward;
    bool is_done;
};

class SimpleEnv {
   public:
    enum class Action { Right, Left, Up, Down, Size = 4 };

    explicit SimpleEnv(const std::shared_ptr<Maze>& maze);

    SimpleEnv();

    virtual State Reset();

    virtual void Render();

    virtual Observation Step(Action action);

    virtual State GetCurrentState() const;

    virtual int NumberOfStates() const;

    virtual State SampleState();

    virtual int NumberOfActions() const;

    virtual Action SampleAction();

    std::shared_ptr<Maze> maze_;

   protected:
    State current_state_;
    std::mt19937 random_generator;

    virtual bool IsValidForStep(State state, std::vector<char> valid_values);

    virtual State GetNextState(State state, Action action);

    virtual Reward GetRewardForAction(State state, Action action);
};

class SwitchEnv : protected SimpleEnv {
   public:
    enum class Action { Right, Left, Up, Down, Switch, Size = 5 };

   protected:
    virtual bool IsValidForStep(State state, std::vector<char> valid_values);

    virtual State GetNextState(State state, Action action);

    virtual Reward GetRewardForAction(State state, Action action);
};