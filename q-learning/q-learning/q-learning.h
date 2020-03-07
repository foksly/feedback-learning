#pragma once

#include <memory>
#include "../environments/grid_world.h"

class QTable {
   public:
    QTable(const SimpleEnv& env, double learning_rate, double discounting_rate);

    explicit QTable(const SimpleEnv& env);

    virtual void SetLearningRate(double learning_rate);

    virtual void SetDiscountingRate(double discounting_rate);

    virtual void UpdateQValue(State state, SimpleEnv::Action action, State new_state,
                              Reward reward);

    SimpleEnv::Action GetBestAction(State state);

    virtual bool AllQValuesEqual(State state);

    virtual void Reset();

    virtual void Render(int n_cols, const Maze& maze);

   protected:
    std::vector<std::vector<double>> qtable_;

    double learning_rate_;
    double discounting_rate_;
};

struct SwitchEnvState {
    State state;
    int n_switches;
};

class SwitchQTable {
   public:
    explicit SwitchQTable(const SwitchEnv& env);

    void UpdateQValue(SwitchEnvState state, SwitchEnv::Action action, SwitchEnvState new_state,
                      Reward reward);

    SwitchEnv::Action GetBestAction(SwitchEnvState state);

    void SetLearningRate(double learning_rate);

    void SetDiscountingRate(double discounting_rate);

    bool AllQValuesEqual(SwitchEnvState state);

    void Render(int n_cols, const Maze& maze);

   protected:
    std::vector<std::vector<std::vector<double>>> qtable_;

    double learning_rate_;
    double discounting_rate_;

    int n_switches_;
    int max_switches_;
    std::vector<State> key_order_;
};

class Epsilon {
   public:
    explicit Epsilon(double value_) : value(value_) {}
    double value = 0.5;

    virtual void Update(int episode){};
    virtual ~Epsilon(){};
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

QTable TrainAutoSwitch1dState(int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon);

SwitchQTable TrainAutoSwitch2dState(int n_episodes, int max_steps,
                                    std::shared_ptr<Epsilon> epsilon);