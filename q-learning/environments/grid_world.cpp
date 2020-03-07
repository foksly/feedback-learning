#include "grid_world.h"

Maze::Maze(int n_rows, int n_cols, State start, State end)
    : n_rows_(n_rows), n_cols_(n_cols), maze_(n_rows * n_cols, kGrid), start_(start), end_(end) {
    maze_[start] = kStart;
    maze_[end] = kEnd;

    value2reward[kGrid] = 0;
    value2reward[kStart] = 0;
    value2reward[kEnd] = 1;
    value2reward[kKey] = 0;
}

Maze::Maze(int size, State start, State end) : Maze(size, size, start, end) {}

Maze::Maze(int num_rows, int num_cols, std::pair<int, int> start, std::pair<int, int> end)
    : Maze(num_rows, num_cols, num_cols * start.first + start.second,
           num_cols * end.first + end.second) {}

Maze::Maze(int size, std::pair<int, int> start, std::pair<int, int> end)
    : Maze(size, size, start, end) {}

char Maze::operator[](State state) const { return maze_[state]; }

char& Maze::operator[](State state) { return maze_[state]; }

size_t Maze::Size() const { return maze_.size(); }

int Maze::NumberOfRows() const { return n_rows_; }

int Maze::NumberOfCols() const { return n_cols_; }

std::vector<char> Maze::GetValidForStepValues() const { return {kGrid, kKey, kStart, kEnd}; }

Reward Maze::GetRewardInState(State state) { return value2reward[maze_[state]]; }

State Maze::GetStartState() const { return start_; }

State Maze::GetEndState() const { return end_; }

State Maze::ConvertCoordinateToState(std::pair<int, int> coordinate) const {
    return n_cols_ * coordinate.first + coordinate.second;
};

std::pair<int, int> Maze::ConvertStateToCoordinate(State state) const {
    return {state / n_cols_, state % n_cols_};
}

void Maze::ChangeRewardValue(char key, Reward reward) { value2reward[key] = reward; }

SimpleEnv::SimpleEnv(const std::shared_ptr<Maze>& maze)
    : maze_(maze), current_state_(maze->GetStartState()), random_generator(time(0)) {}

SimpleEnv::SimpleEnv() : SimpleEnv(std::make_shared<Maze>(5, 0, 24)) {}

State SimpleEnv::Reset() {
    current_state_ = maze_->GetStartState();
    return current_state_;
}

void SimpleEnv::Render() {
    for (int index = 0; index < static_cast<int>(maze_->Size()); ++index) {
        if (index % maze_->NumberOfRows() == 0) {
            std::cout << "\n";
        }
        if (index == current_state_) {
            std::cout << "(C, " << maze_->GetRewardInState(current_state_) << ") ";
        } else {
            std::cout << "(" << (*maze_)[index] << ", " << maze_->GetRewardInState(index) << ") ";
        }
    }
    std::cout << "\n";
}

Observation SimpleEnv::Step(Action action) {
    State next_state = GetNextState(current_state_, action);
    Reward reward = GetRewardForAction(current_state_, action);
    bool is_done = false;
    if (next_state == maze_->GetEndState()) {
        is_done = true;
    }
    current_state_ = next_state;
    return {next_state, reward, is_done};
}

int SimpleEnv::NumberOfStates() const { return static_cast<int>(maze_->Size()); }

State SimpleEnv::SampleState() {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(maze_->Size()) - 1);
    return dist(random_generator);
}

int SimpleEnv::NumberOfActions() const { return static_cast<int>(Action::Size); }

SimpleEnv::Action SimpleEnv::SampleAction() {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(Action::Size) - 1);
    return static_cast<Action>(dist(random_generator));
}

State SimpleEnv::GetCurrentState() const { return current_state_; }

bool SimpleEnv::IsValidForStep(State state, std::vector<char> valid_values) {
    bool is_valid = false;
    for (auto value : valid_values) {
        is_valid |= (*maze_)[state] == value;
    }
    return is_valid;
}

State SimpleEnv::GetNextState(State state, Action action) {
    std::pair<int, int> state_coordinates = maze_->ConvertStateToCoordinate(state);
    if (action == Action::Right && state_coordinates.second < maze_->NumberOfCols() - 1) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first, state_coordinates.second + 1});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Left && state_coordinates.second > 0) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first, state_coordinates.second - 1});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Up && state_coordinates.first > 0) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first - 1, state_coordinates.second});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Down && state_coordinates.first < maze_->NumberOfRows() - 1) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first + 1, state_coordinates.second});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    return state;
}

Reward SimpleEnv::GetRewardForAction(State state, Action action) {
    State next_state = GetNextState(state, action);
    return maze_->GetRewardInState(next_state);
}

SwitchEnv::SwitchEnv(const std::shared_ptr<Maze>& maze, std::vector<State> key_order)
    : SimpleEnv(maze),
      n_swithches_(0),
      max_switches_(static_cast<int>(key_order.size())),
      key_order_(key_order),
      correct_switches_(key_order.size(), false) {}

State SwitchEnv::Reset() {
    current_state_ = maze_->GetStartState();
    n_swithches_ = 0;
    correct_switches_.assign(correct_switches_.size(), false);
    return current_state_;
}

State SwitchEnv::GetNextState(State state, Action action) {
    std::pair<int, int> state_coordinates = maze_->ConvertStateToCoordinate(state);
    if (action == Action::Right && state_coordinates.second < maze_->NumberOfCols() - 1) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first, state_coordinates.second + 1});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Left && state_coordinates.second > 0) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first, state_coordinates.second - 1});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Up && state_coordinates.first > 0) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first - 1, state_coordinates.second});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    if (action == Action::Down && state_coordinates.first < maze_->NumberOfRows() - 1) {
        State next_state = maze_->ConvertCoordinateToState(
            {state_coordinates.first + 1, state_coordinates.second});
        if (IsValidForStep(next_state, maze_->GetValidForStepValues())) {
            return next_state;
        }
    }
    return state;
}

Reward SwitchEnv::GetRewardForAction(State state, Action action) {
    State next_state = GetNextState(state, action);
    if (action == Action::Switch && n_swithches_ < max_switches_ &&
        key_order_[n_swithches_] == state) {
        return 0;
    }
    if (next_state == maze_->GetEndState()) {
        bool all_correct = true;
        for (auto value : correct_switches_) {
            all_correct &= value;
        }
        if (all_correct) {
            return 1;
        }
    }
    return 0;
    // return maze_->GetRewardInState(next_state);
}

Observation SwitchEnv::Step(Action action) {
    Reward reward = GetRewardForAction(current_state_, action);
    if (action == Action::Switch && n_swithches_ < max_switches_) {
        if (key_order_[n_swithches_] == current_state_) {
            correct_switches_[n_swithches_] = true;
        }
        ++n_swithches_;
    }
    State next_state = GetNextState(current_state_, action);

    bool is_done = false;
    if (next_state == maze_->GetEndState()) {
        is_done = true;
    }
    current_state_ = next_state;
    return {next_state, reward, is_done};
}

int SwitchEnv::NumberOfActions() const { return static_cast<int>(Action::Size); }

SwitchEnv::Action SwitchEnv::SampleAction() {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(SwitchEnv::Action::Size) - 1);
    return static_cast<SwitchEnv::Action>(dist(random_generator));
}

int SwitchEnv::GetNumberOfSwitches() const { return n_swithches_; }

std::vector<State> SwitchEnv::GetKeyOrder() const { return key_order_; }

void SwitchEnv::Render() {
    std::cout << "Number of switches: " << n_swithches_ << "\n";
    for (int index = 0; index < static_cast<int>(maze_->Size()); ++index) {
        if (index % maze_->NumberOfRows() == 0) {
            std::cout << "\n";
        }
        if (index == current_state_) {
            std::cout << "C ";
        } else {
            std::cout << (*maze_)[index] << " ";
        }
    }
    std::cout << "\n";
}