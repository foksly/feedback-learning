#include "q-learning.h"

QTable::QTable(const SimpleEnv& env, double learning_rate, double discounting_rate)
    : qtable_(env.NumberOfStates(), std::vector<double>(env.NumberOfActions(), 0.0)),
      learning_rate_(learning_rate),
      discounting_rate_(discounting_rate) {}

QTable::QTable(const SimpleEnv& env) : QTable(env, 0.5, 0.7) {}

void QTable::SetLearningRate(double learning_rate) { learning_rate_ = learning_rate; }

void QTable::SetDiscountingRate(double discounting_rate) { discounting_rate_ = discounting_rate; }

void QTable::UpdateQValue(State state, SimpleEnv::Action action, State new_state, Reward reward) {
    auto max_new_state_qvalue =
        *std::max_element(qtable_[new_state].begin(), qtable_[new_state].end());

    qtable_[state][static_cast<int>(action)] =
        qtable_[state][static_cast<int>(action)] +
        learning_rate_ * (reward + discounting_rate_ * max_new_state_qvalue -
                          qtable_[state][static_cast<int>(action)]);
}

SimpleEnv::Action QTable::GetBestAction(State state) {
    auto iter = std::max_element(qtable_[state].begin(), qtable_[state].end());
    return static_cast<SimpleEnv::Action>(iter - qtable_[state].begin());
}

bool QTable::AllQValuesEqual(State state) {
    auto value = qtable_[state][0];
    for (auto v : qtable_[state]) {
        if (v != value) {
            return false;
        }
    }
    return true;
}

void QTable::Reset() {
    for (auto state_vector : qtable_) {
        std::fill(state_vector.begin(), state_vector.end(), 0.0);
    }
}

void QTable::Render(int n_cols, const Maze& maze) {
    std::vector<char> direction = {'R', 'L', 'U', 'D'};
    for (int index = 0; index < static_cast<int>(qtable_.size()); ++index) {
        if (index % n_cols == 0) {
            std::cout << "\n";
        }
        if (maze[index] == maze.kWall) {
            std::cout << "\033[1;31mW\033[0m"
                      << " ";
        } else if (maze[index] == maze.kStart) {
            std::cout << "\033[1;31mS\033[0m"
                      << " ";
        } else if (maze[index] == maze.kEnd) {
            std::cout << "\033[1;31mE\033[0m"
                      << " ";
        }

        else if (maze[index] == maze.kKey) {
            std::cout << "\033[1;31mK\033[0m"
                      << " ";
        } else if (AllQValuesEqual(index)) {
            std::cout << "- ";
        }

        else {
            std::cout << direction[static_cast<int>(GetBestAction(index))] << " ";
        }
    }
    std::cout << "\n";
}

void EpsilonWithDecay::Update(int episode) {
    value = min_epsilon + (max_epsilon - min_epsilon) * std::exp(-decay_rate * episode);
}

QTable Train(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon) {
    Maze maze(10, 0, 99);
    SimpleEnv env(std::make_shared<Maze>(maze));
    QTable qtable(env);
    std::mt19937 random_generator(time(0));
    std::uniform_real_distribution<> dist(0, 1);

    env.Render();

    for (int episode = 0; episode < n_episodes; ++episode) {
        State state = env.Reset();

        int n_steps = 0;
        bool is_done = false;

        while (n_steps < max_steps && !is_done) {
            ++n_steps;

            SimpleEnv::Action action = qtable.GetBestAction(state);
            if (dist(random_generator) < epsilon->value || qtable.AllQValuesEqual(state)) {
                action = env.SampleAction();
            }
            Observation observation = env.Step(action);
            qtable.UpdateQValue(state, action, observation.state, observation.reward);
            is_done = observation.is_done;
            state = observation.state;
        }
        epsilon->Update(episode);
        qtable.Render(maze.NumberOfCols(), maze);
    }

    return qtable;
}

QTable TrainAutoSwitch1dState(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon) {
    // Maze maze(5, 0, 24);
    SimpleEnv env(std::make_shared<Maze>(5, 0, 24));
    (*env.maze_)[4] = env.maze_->kKey;

    QTable qtable(env);
    std::mt19937 random_generator(time(0));
    std::uniform_real_distribution<> dist(0, 1);

    env.Render();

    for (int episode = 0; episode < n_episodes; ++episode) {
        State state = env.Reset();

        int n_steps = 0;
        bool is_done = false;

        env.maze_->ChangeRewardValue(env.maze_->kEnd, 0);

        while (n_steps < max_steps && !is_done) {
            ++n_steps;

            SimpleEnv::Action action = qtable.GetBestAction(state);
            if (dist(random_generator) < epsilon->value || qtable.AllQValuesEqual(state)) {
                action = env.SampleAction();
            }

            Observation observation = env.Step(action);
            if ((*env.maze_)[observation.state] == env.maze_->kKey) {
                env.maze_->ChangeRewardValue(env.maze_->kEnd, 1);
            }

            qtable.UpdateQValue(state, action, observation.state, observation.reward);
            is_done = observation.is_done;
            state = observation.state;
        }
        epsilon->Update(episode);
        std::cout << "Episode: " << episode << "\n";
        env.Render();
        qtable.Render(env.maze_->NumberOfCols(), *env.maze_);
    }

    return qtable;
}

SwitchQTable::SwitchQTable(const SwitchEnv& env)
    : qtable_(env.max_switches_ + 1,
              std::vector<std::vector<double>>(env.NumberOfStates(),
                                               std::vector<double>(env.NumberOfActions(), 0.0))),
      learning_rate_(0.5),
      discounting_rate_(0.9) {}

void SwitchQTable::UpdateQValue(SwitchEnvState state, SwitchEnv::Action action,
                                SwitchEnvState new_state, Reward reward) {
    // std::cout << "INSIDE UPD\n";
    // std::cout << qtable_.size() << "\n";
    auto max_new_state_qvalue =
        *std::max_element(qtable_[new_state.n_switches][new_state.state].begin(),
                          qtable_[new_state.n_switches][new_state.state].end());

    qtable_[state.n_switches][state.state][static_cast<int>(action)] =
        qtable_[state.n_switches][state.state][static_cast<int>(action)] +
        learning_rate_ * (reward + discounting_rate_ * max_new_state_qvalue -
                          qtable_[state.n_switches][state.state][static_cast<int>(action)]);
}

SwitchEnv::Action SwitchQTable::GetBestAction(SwitchEnvState state) {
    auto iter = std::max_element(qtable_[state.n_switches][state.state].begin(),
                                 qtable_[state.n_switches][state.state].end());
    return static_cast<SwitchEnv::Action>(iter - qtable_[state.n_switches][state.state].begin());
}

void SwitchQTable::SetLearningRate(double learning_rate) { learning_rate_ = learning_rate; }

void SwitchQTable::SetDiscountingRate(double discounting_rate) {
    discounting_rate_ = discounting_rate;
}

bool SwitchQTable::AllQValuesEqual(SwitchEnvState state) {
    auto value = qtable_[state.n_switches][state.state][0];
    for (auto v : qtable_[state.n_switches][state.state]) {
        if (v != value) {
            return false;
        }
    }
    return true;
}

void SwitchQTable::Render(int n_cols, const Maze& maze) {
    std::vector<char> direction = {'R', 'L', 'U', 'D', 'S'};

    for (int n_switch = 0; n_switch < static_cast<int>(qtable_.size()); ++n_switch) {
        std::cout << "\n";
        std::cout << "Number of switches: " << n_switch;
        for (int index = 0; index < static_cast<int>(qtable_[n_switch].size()); ++index) {
            if (index % n_cols == 0) {
                std::cout << "\n";
            }
            char current_direction = direction[static_cast<int>(GetBestAction({index, n_switch}))];
            if (AllQValuesEqual({index, n_switch})) {
                current_direction = '-';
            }
            if (maze[index] == maze.kWall) {
                std::cout << "\033[1;31mW\033[0m" << current_direction << " ";
            } else if (maze[index] == maze.kStart) {
                std::cout << "\033[1;31mB\033[0m" << current_direction << " ";
            } else if (maze[index] == maze.kEnd) {
                std::cout << "\033[1;31mE\033[0m" << current_direction << " ";
            }

            else if (maze[index] == maze.kKey) {
                std::cout << "\033[1;31mK\033[0m" << current_direction << " ";
            } else if (AllQValuesEqual({index, n_switch})) {
                std::cout << "-  ";
            }

            else {
                std::cout << current_direction << "  ";
            }
        }
        std::cout << "\n";
    }
}

SwitchQTable TrainAutoSwitch2dState(int n_episodes, int max_steps,
                                    std::shared_ptr<Epsilon> epsilon) {
    SwitchEnv env(std::make_shared<Maze>(5, 20, 24), {2});
    for (auto s : env.GetKeyOrder()) {
        (*env.maze_)[s] = env.maze_->kKey;
    }

    SwitchQTable qtable(env);
    std::mt19937 random_generator(time(0));
    std::uniform_real_distribution<> dist(0, 1);

    env.Render();

    for (int episode = 0; episode < n_episodes; ++episode) {
        State state = env.Reset();

        int n_steps = 0;
        bool is_done = false;

        int total_reward = 0;

        while (n_steps < max_steps && !is_done) {
            ++n_steps;

            SwitchEnv::Action action = qtable.GetBestAction({state, env.GetNumberOfSwitches()});
            if (dist(random_generator) < epsilon->value ||
                qtable.AllQValuesEqual({state, env.GetNumberOfSwitches()})) {
                action = env.SampleAction();
                while (action == SwitchEnv::Action::Switch) {
                    action = env.SampleAction();
                }
            }

            if (state == env.GetKeyOrder()[0] && env.GetNumberOfSwitches() < env.max_switches_) {
                action = SwitchEnv::Action::Switch;
            }

            int current_switches = env.GetNumberOfSwitches();

            Observation observation = env.Step(action);

            total_reward += observation.reward;

            qtable.UpdateQValue({state, current_switches}, action,
                                {observation.state, env.GetNumberOfSwitches()}, observation.reward);
            is_done = observation.is_done;
            state = observation.state;
        }
        epsilon->Update(episode);
        std::cout << "=======================\n";
        std::cout << "Episode: " << episode << "\n";
        std::cout << "Total reward: " << total_reward << "\n";
        env.Render();
        qtable.Render(env.maze_->NumberOfCols(), *env.maze_);
        std::cout << "CORRECT SWITCHES: ";
        for (auto corr : env.correct_switches_) {
            if (corr) {
                std::cout << "T ";
            } else {
                std::cout << "F ";
            }
        }
        std::cout << "\n";
    }

    return qtable;
}