#pragma once

#include "../environments/grid_world.h"
#include <memory>

class QTable {
   public:
    QTable(const Environment& env, double learning_rate, double discounting_rate);

    explicit QTable(const Environment& env);

    virtual void SetLearningRate(double learning_rate);

    virtual void SetDiscountingRate(double discounting_rate);

    virtual void UpdateQValue(State state, Action action, State new_state, Reward reward);

    virtual Action GetBestAction(State state);

    virtual void Reset();

    virtual void Render(int n_cols, const Maze& maze);

   private:
    std::vector<std::vector<double>> qtable_;

    double learning_rate_;
    double discounting_rate_;
};

class Epsilon {
public:
    explicit Epsilon(double value_) : value(value_) {}
    double value = 0.5;

    virtual void Update(int episode) {};
    virtual ~Epsilon() {};
};

class EpsilonWithDecay : public Epsilon {
public:
    explicit EpsilonWithDecay(double value) : Epsilon(value) {}
    double max_epsilon = 1.0;
    double min_epsilon = 0.01;
    double decay_rate = 0.01;

    virtual void Update(int episode);

};

QTable Train(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon);

QTable TrainModelProblemN1(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon);

QTable TrainModelProblemN2(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon);