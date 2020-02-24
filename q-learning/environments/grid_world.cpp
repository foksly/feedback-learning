// #pragma once

#include "grid_world.h"

Maze::Maze(int n_rows, int n_cols, State start, State end)
    : n_rows_(n_rows), n_cols_(n_cols), maze_(n_rows * n_cols, kGrid), start_(start), end_(end) {
    maze_[start] = kStart;
    maze_[end] = kEnd;
    // assert kStart != kend

    value_to_reward[kGrid] = -1;
    value_to_reward[kStart] = -1;
    value_to_reward[kEnd] = 100;
    value_to_reward[kKey] = 1;
}

Maze::Maze(int size, State start, State end) : Maze(size, size, start, end) {}

Maze::Maze(int num_rows, int num_cols, std::pair<int, int> start, std::pair<int, int> end)
    : Maze(num_rows, num_cols, num_cols * start.first + start.second,
           num_cols * end.first + end.second) {}

Maze::Maze(int size, std::pair<int, int> start, std::pair<int, int> end)
    : Maze(size, size, start, end) {}

char Maze::operator[](State state) { return maze_[state]; };

size_t Maze::Size() const { return maze_.size(); }

int Maze::NumberOfRows() const { return n_rows_; };

int Maze::NumberOfCols() const { return n_cols_; };

std::vector<char> Maze::GetValidForStepValues() const { return {kGrid, kKey, kStart, kEnd}; }

Reward Maze::GetRewardInState(State state) { return value_to_reward[maze_[state]]; }

State Maze::GetStartState() const { return start_; }

State Maze::GetEndState() const { return end_; }

Environment::Environment(Maze maze)
    : maze_(maze), current_state_(maze.GetStartState()), random_generator(151343) {}

Environment::Environment() : Environment({5, 0, 24}) {}

State Environment::Reset() {
    current_state_ = maze_.GetStartState();
    return current_state_;
}

void Environment::Render() {
    for (int index = 0; index < static_cast<int>(maze_.Size()); ++index) {
        if (index % maze_.NumberOfRows() == 0) {
            std::cout << "\n";
        }
        if (index == current_state_) {
            std::cout << "C"
                      << " ";
        } else {
            std::cout << maze_[index] << " ";
        }
    }
    std::cout << "\n";
}

Observation Environment::Step(Action action) {
    State next_state = GetNextState(current_state_, action);
    Reward reward = GetRewardForAction(current_state_, action);
    bool is_done = false;
    if (next_state == maze_.GetEndState()) {
        is_done = true;
    }
    current_state_ = next_state;
    return {next_state, reward, is_done};
}

int Environment::NumberOfStates() { return static_cast<int>(maze_.Size()); }

State Environment::SampleState() {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(maze_.Size()) - 1);
    return dist(random_generator);
}

int Environment::NumberOfActions() { return static_cast<int>(Action::Size); }

Action Environment::SampleAction() {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(Action::Size) - 1);
    return static_cast<Action>(dist(random_generator));
}

State Environment::GetCurrentState() const { return current_state_; }

State Environment::ConvertCoordinateToState(std::pair<int, int> coordinate) {
    return maze_.NumberOfCols() * coordinate.first + coordinate.second;
};

std::pair<int, int> Environment::ConvertStateToCoordinate(State state) {
    return {state / maze_.NumberOfCols(), state % maze_.NumberOfCols()};
}

bool Environment::IsValidForStep(State state, std::vector<char> valid_values) {
    bool is_valid = false;
    for (auto value : valid_values) {
        is_valid |= maze_[state] == value;
    }
    return is_valid;
}

State Environment::GetNextState(State state, Action action) {
    std::pair<int, int> state_coordinates = ConvertStateToCoordinate(state);
    if (action == Action::Right && state_coordinates.second < maze_.NumberOfCols() - 1) {
        State next_state =
            ConvertCoordinateToState({state_coordinates.first, state_coordinates.second + 1});
        if (IsValidForStep(next_state, maze_.GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Left && state_coordinates.second > 0) {
        State next_state =
            ConvertCoordinateToState({state_coordinates.first, state_coordinates.second - 1});
        if (IsValidForStep(next_state, maze_.GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Up && state_coordinates.first > 0) {
        State next_state =
            ConvertCoordinateToState({state_coordinates.first - 1, state_coordinates.second});
        if (IsValidForStep(next_state, maze_.GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Down && state_coordinates.first < maze_.NumberOfRows() - 1) {
        State next_state =
            ConvertCoordinateToState({state_coordinates.first + 1, state_coordinates.second});
        if (IsValidForStep(next_state, maze_.GetValidForStepValues())) {
            return next_state;
        }
    }
    return state;
}

Reward Environment::GetRewardForAction(State state, Action action) {
    State next_state = GetNextState(state, action);
    return maze_.GetRewardInState(next_state);
}