#include "q-learning.h"

QTable::QTable(const Environment& env, double learning_rate, double discounting_rate)
    : qtable_(env.NumberOfStates(), std::vector<double>(env.NumberOfActions(), 0.0)),
      learning_rate_(learning_rate),
      discounting_rate_(discounting_rate) {}

QTable::QTable(const Environment& env) : QTable(env, 0.5, 0.7) {}

void QTable::SetLearningRate(double learning_rate) { learning_rate_ = learning_rate; }

void QTable::SetDiscountingRate(double discounting_rate) { discounting_rate_ = discounting_rate; }

void QTable::UpdateQValue(State state, Action action, State new_state, Reward reward) {
    auto max_new_state_qvalue =
        *std::max_element(qtable_[new_state].begin(), qtable_[new_state].end());

    qtable_[state][static_cast<int>(action)] =
        qtable_[state][static_cast<int>(action)] +
        learning_rate_ * (reward + discounting_rate_ * max_new_state_qvalue -
                          qtable_[state][static_cast<int>(action)]);
}

Action QTable::GetBestAction(State state) {
    auto iter = std::max_element(qtable_[state].begin(), qtable_[state].end());
    return static_cast<Action>(iter - qtable_[state].begin());
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
            std::cout << "W"
                      << " ";
        } 
        else if (maze[index] == maze.kStart) {
            std::cout << "S" << " ";
        }
        else if (maze[index] == maze.kEnd) {
            std::cout << "E" << " ";
        }

        else if (maze[index] == maze.kKey) {
            std::cout << "K" << " ";
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
    Maze maze(5, 0, 8);
    Environment env(std::make_shared<Maze>(maze));
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

            Action action = qtable.GetBestAction(state);
            if (dist(random_generator) < epsilon->value) {
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

QTable TrainModelProblemN1(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon) {
    Maze maze(5, 0, 8);
    Environment env(std::make_shared<Maze>(maze));
    (*env.maze_)[22] = env.maze_->kKey;

    QTable qtable(env);
    std::mt19937 random_generator(time(0));
    std::uniform_real_distribution<> dist(0, 1);

    env.Render();

    for (int episode = 0; episode < n_episodes; ++episode) {
        State state = env.Reset();


        int n_steps = 0;
        bool is_done = false;

        env.maze_->ChangeRewardValue(maze.kEnd, 0);

        std::cout << "Start of episode\n";
        env.Render();

        int total_reward = 0;

        while (n_steps < max_steps && !is_done) {
            ++n_steps;

            Action action = qtable.GetBestAction(state);
            if (dist(random_generator) < epsilon->value) {
                action = env.SampleAction();
            }

            Observation observation = env.Step(action);
            if ((*env.maze_)[observation.state] == maze.kKey) {
                env.maze_->ChangeRewardValue(maze.kEnd, 1);
            }

            total_reward += observation.reward;

            qtable.UpdateQValue(state, action, observation.state, observation.reward);
            is_done = observation.is_done;
            state = observation.state;

        }
        epsilon->Update(episode);
        std::cout << "==================\n";
        std::cout << "Episode: " << episode + 1 << " | Total reward: " << total_reward;
        qtable.Render(maze.NumberOfCols(), *env.maze_);
        env.Render();
        std::cout << "==================\n";
    }

    return qtable;
}


QTable TrainModelProblemN2(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon) {
    Maze maze(5, 0, 8);
    Environment env(std::make_shared<Maze>(maze));
    (*env.maze_)[22] = env.maze_->kKey;

    std::vector<QTable> qtable(2, QTable(env));
    std::mt19937 random_generator(time(0));
    std::uniform_real_distribution<> dist(0, 1);

    env.Render();

    for (int episode = 0; episode < n_episodes; ++episode) {
        State state = env.Reset();


        int n_steps = 0;
        bool is_done = false;

        int key_taken = 0;

        std::cout << "Start of episode\n";
        env.Render();

        int reward = 0;

        while (n_steps < max_steps && !is_done) {
            ++n_steps;

            Action action = qtable[key_taken].GetBestAction(state);
            if (dist(random_generator) < epsilon->value) {
                action = env.SampleAction();
            }

            Observation observation = env.Step(action);
            if ((*env.maze_)[observation.state] == maze.kKey) {
                key_taken = 1;
            }
            
            is_done = observation.is_done;
            if (is_done && key_taken == 1) {
                reward = 1;
            }

            qtable[key_taken].UpdateQValue(state, action, observation.state, reward);
            
            state = observation.state;

        }
        epsilon->Update(episode);
        std::cout << "==================\n";
        std::cout << "Episode: " << episode + 1 << " | Total reward: " << reward << "\n";
        std::cout << "Table 0";
        qtable[0].Render(maze.NumberOfCols(), *env.maze_);
        std::cout << "\n Table 1";
        qtable[1].Render(maze.NumberOfCols(), *env.maze_);
        env.Render();
        std::cout << "==================\n";
    }

    return qtable[0];
}
