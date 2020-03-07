#include "q-learning_vis.h"

sf::RectangleShape GetRect(int size, std::pair<int, int> position, sf::Color fill_color,
                           sf::Color outline_color, int thickness) {
    sf::RectangleShape rectangle;
    rectangle.setSize(sf::Vector2f(size, size));
    rectangle.setOutlineColor(outline_color);
    rectangle.setFillColor(fill_color);
    rectangle.setOutlineThickness(thickness);
    rectangle.setPosition(position.first, position.second);
    return rectangle;
}

sf::Text GetText(std::string message, const sf::Font& font, int char_size,
                 std::pair<int, int> position, sf::Color color) {
    sf::Text text(message, font);
    text.setCharacterSize(char_size);
    text.setFillColor(color);
    text.setPosition(position.first, position.second);
    return text;
}

void DrawGrid(sf::RenderWindow* window, const Maze& maze) {
    sf::Font font;
    font.loadFromFile(kFontPath);

    int thickness = 1;
    for (int row = 0; row < maze.NumberOfRows(); ++row) {
        for (int col = 0; col < maze.NumberOfCols(); ++col) {
            sf::RectangleShape rectangle =
                GetRect(kSquareSize - 2 * thickness,
                        {col * kSquareSize + thickness, row * kSquareSize + thickness},
                        sf::Color::White, sf::Color::Black, thickness);
            if (maze[maze.ConvertCoordinateToState({row, col})] == maze.kStart) {
                rectangle.setFillColor(sf::Color(154, 205, 50));
                sf::Text text = GetText("S", font, 50, {col * kSquareSize + 16, row * kSquareSize});
                window->draw(rectangle);
                window->draw(text);
            } else if (maze[maze.ConvertCoordinateToState({row, col})] == maze.kEnd) {
                rectangle.setFillColor(sf::Color(220, 20, 60));
                sf::Text text = GetText("E", font, 50, {col * kSquareSize + 16, row * kSquareSize});
                window->draw(rectangle);
                window->draw(text);
            } else if (maze[maze.ConvertCoordinateToState({row, col})] == maze.kWall) {
                rectangle.setFillColor(sf::Color::Blue);
                window->draw(rectangle);
            } else {
                window->draw(rectangle);
            }
        }
    }
}

void DrawCurrentState(sf::RenderWindow* window, SimpleEnv* env) {
    auto state_position = env->maze_->ConvertStateToCoordinate(env->GetCurrentState());
    int thickness = 1;
    sf::RectangleShape rectangle = GetRect(kSquareSize - 2 * thickness,
                                           {state_position.second * kSquareSize + thickness,
                                            state_position.first * kSquareSize + thickness},
                                           sf::Color::Red, sf::Color::Black, thickness);

    sf::Font font;
    font.loadFromFile(kFontPath);
    sf::Text text =
        GetText("A", font, 50,
                {state_position.second * kSquareSize + 16, state_position.first * kSquareSize});
    window->draw(rectangle);
    window->draw(text);
}

void VisualizeQLearning(std::pair<int, int> maze_size, State start, State end, int speed,
                        int n_episodes, int max_steps, std::shared_ptr<Epsilon> epsilon) {
    Maze maze = Maze(maze_size.first, maze_size.second, start, end);
    SimpleEnv env(std::make_shared<Maze>(maze));

    QTable qtable(env);

    sf::RenderWindow window(
        sf::VideoMode(maze.NumberOfCols() * kSquareSize, maze.NumberOfRows() * kSquareSize),
        "Grid World Q-learning", sf::Style::Close);

    std::mt19937 random_generator(1531413);
    std::uniform_real_distribution<> dist(0, 1);

    bool is_started = false;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::MouseButtonPressed && !is_started) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    (*env.maze_)[env.maze_->ConvertCoordinateToState(
                        {event.mouseButton.y / kSquareSize, event.mouseButton.x / kSquareSize})] =
                        env.maze_->kWall;
                }
                if (event.mouseButton.button == sf::Mouse::Right) {
                    (*env.maze_)[env.maze_->ConvertCoordinateToState(
                        {event.mouseButton.y / kSquareSize, event.mouseButton.x / kSquareSize})] =
                        env.maze_->kGrid;
                }
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    is_started = true;
                }
            }
        }
        if (!is_started) {
            window.clear(sf::Color::White);
            DrawGrid(&window, *env.maze_);
            window.display();
        } else {
            for (int episode = 0; episode < n_episodes; ++episode) {
                State state = env.Reset();

                int n_steps = 0;
                bool is_done = false;

                std::cout << "Episode: " << episode << "\n";

                while (n_steps < max_steps && !is_done) {
                    ++n_steps;
                    sf::Event event;
                    while (window.pollEvent(event)) {
                        if (event.type == sf::Event::Closed) {
                            window.close();
                            return;
                        }
                    }
                    window.clear(sf::Color::White);
                    DrawGrid(&window, *env.maze_);
                    DrawCurrentState(&window, &env);
                    window.display();
                    sf::sleep(sf::milliseconds(speed));

                    SimpleEnv::Action action = qtable.GetBestAction(state);
                    if (dist(random_generator) < epsilon->value) {
                        action = env.SampleAction();
                    }
                    Observation observation = env.Step(action);
                    qtable.UpdateQValue(state, action, observation.state, observation.reward);
                    is_done = observation.is_done;
                    state = observation.state;
                }
                epsilon->Update(episode);
                qtable.Render(env.maze_->NumberOfCols(), *env.maze_);
            }
            is_started = false;
        }
    }
}