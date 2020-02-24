#include "environments/grid_world.h"
#include "q-learning/q-learning.h"

int main() {
    Epsilon eps;
    auto qtable = Train(30, 40, eps);
    return 0;
}