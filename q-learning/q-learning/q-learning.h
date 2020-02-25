#pragma once

#include "../environments/grid_world.h"

class QTable {
   public:
    QTable(Environment env, double learning_rate, double discounting_rate);

    explicit QTable(Environment env);

    void SetLearningRate(double learning_rate);

    void SetDiscountingRate(double discounting_rate);

    void UpdateQValue(State state, Action action, State new_state, Reward reward);

    Action GetBestAction(State state);

    void Reset();

    void Render(int n_cols, const Maze& maze);

   private:
    std::vector<std::vector<double>> qtable_;
    double learning_rate_;
    double discounting_rate_;
};

struct Epsilon {
    double value = 1.0;
    double max_epsilon = 1.0;
    double min_epsilon = 0.01;
    double decay_rate = 0.01;

    void Update(int episode);
};

QTable Train(int n_episodes, int max_steps, Epsilon epsilon);