#include "q-learning/q-learning.h"
#include "vis/q-learning_vis.h"

int main() {
    std::pair<int, int> grid_size{5, 7};
    State start = 0;
    State end = 34;
    int speed = 0;
    int n_episodes = 300;
    int max_steps = 400;
    auto eps = std::make_shared<EpsilonWithDecay>(0.7);
    // auto eps = std::make_shared<Epsilon>(0.4);

    // Train(n_episodes, max_steps, eps);
    // TrainAutoSwitch1dState(n_episodes, max_steps, eps);
    // TrainAutoSwitch2dState(n_episodes, max_steps, eps);
    TrainBackAndForth(n_episodes, max_steps, eps);
    // TrainModelProblemN1(n_episodes, max_steps, eps);
    // VisualizeQLearning(grid_size, start, end, speed, n_episodes, max_steps, eps);

    return 0;
}