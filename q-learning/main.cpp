#include "q-learning/q-learning.h"

int main() {
    std::pair<int, int> grid_size{5, 7};
    State start = 0;
    State end = 34;
    int speed = 70;
    int n_episodes = 100;
    int max_steps = 200;
    auto eps = std::make_shared<EpsilonWithDecay>(0.5);
    // auto eps = std::make_shared<Epsilon>(0.3);

    // Train(n_episodes, max_steps, eps);
    TrainModelProblemN1(n_episodes, max_steps, eps);

    return 0;
}