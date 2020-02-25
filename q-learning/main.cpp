#include "vis/q-learning_vis.h"

int main() {
    std::pair<int, int> grid_size{5, 7};
    State start = 0;
    State end = 34;
    int speed = 70;
    int n_episodes = 50;
    int max_steps = 200;
    Epsilon eps;

    VisualizeQLearning(grid_size, start, end, speed, n_episodes, max_steps, eps);

    return 0;
}