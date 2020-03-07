#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Text.hpp>
#include <string>

#include "../q-learning/q-learning.h"

const int kSquareSize = 64;
const std::string kFontPath =
    "/home/foksly/Documents/road-to-nips/feedback-learning/q-learning/vis/"
    "roboto.ttf";

sf::RectangleShape GetRect(int size, std::pair<int, int> position, sf::Color fill_color,
                           sf::Color outline_color, int thickness);

sf::Text GetText(std::string message, const sf::Font& font, int char_size,
                 std::pair<int, int> position, sf::Color color = sf::Color::Black);

void DrawGrid(sf::RenderWindow* window, const Maze& maze);

void DrawCurrentState(sf::RenderWindow* window, SimpleEnv* env);

void VisualizeQLearning(std::pair<int, int> maze_size, State start, State end, int speed,
                        int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon);