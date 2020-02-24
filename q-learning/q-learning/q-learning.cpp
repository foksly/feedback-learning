#include "q-learning.h"

QTable::QTable(Environment env, double learning_rate, double discounting_rate)
    : qtable_(env.NumberOfStates(), std::vector<double>(env.NumberOfActions(), 0.0)),
      learning_rate_(learning_rate),
      discounting_rate_(discounting_rate) {}

QTable::QTable(Environment env) : QTable(env, 0.5, 0.7) {}

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

void QTable::Render(int n_cols) {
    std::vector<char> direction = {'R', 'L', 'U', 'D'};
    for (int index = 0; index < static_cast<int>(qtable_.size()); ++index) {
        if (index % n_cols == 0) {
            std::cout << "\n";
        }
        std::cout << direction[static_cast<int>(GetBestAction(index))] << " ";
    }
    std::cout << "\n";
}

void Epsilon::Update(int episode) {
        value = min_epsilon + (max_epsilon - min_epsilon) * std::exp(-decay_rate * episode);
    }

QTable Train(int n_episodes, int max_steps, Epsilon epsilon) {
    Environment env;
    QTable qtable(env);
    std::mt19937 random_generator(1531413);
    std::uniform_real_distribution<> dist(0, 1);

    for (int episode = 0; episode < n_episodes; ++episode) {
        State state = env.Reset();

        int n_steps = 0;
        bool is_done = false;

        while (n_steps < max_steps && !is_done) {
            Action action = qtable.GetBestAction(state);
            if (dist(random_generator) < epsilon.value) {
                action = env.SampleAction();
            }
            Observation observation = env.Step(action);
            qtable.UpdateQValue(state, action, observation.state, observation.reward);
            is_done = observation.is_done;
            state = observation.state;
        }
        epsilon.Update(episode);
        qtable.Render(env.maze_.NumberOfCols());
    }

    return qtable;
}