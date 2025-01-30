# https://karpathy.github.io/2016/05/31/rl/
import sys
import pygame
import torch
import numpy as np
import ann

sys.path.insert(0, './pong_game')
import pong_game.pong as pong
import pong_game.settings as set


if __name__ == "__main__":
    train_pong = pong.Pong(None, 100)
    train_data = train_pong.main()
    print(train_data)
    del train_pong

    pygame.init()
    screen = pygame.display.set_mode((set.WIDTH, set.HEIGHT))
    pygame.display.set_caption("Ping Pong")

    play_pong = pong.Pong(screen, 10)
    play_pong.main()


if __name__ == "__test__":
    # Generate sample data
    X_train = np.random.rand(1000, 10)
    y_train = np.random.rand(1000, 1)
    X_test = np.random.rand(200, 10)
    y_test = np.random.rand(200, 1)

    # Create data loaders
    train_loader = ann.TrainingData.get_dataloader(X_train, y_train)
    test_loader = ann.TrainingData.get_dataloader(X_test, y_test, shuffle=False)

    # Create neural network instance
    model = ann.NeuralNetwork(input_size=10, output_size=1, hidden_layers=[128, 64], learning_rate=0.01)
    model.summary()

    # Train model
    model.train_model(train_loader, epochs=20)

    # Evaluate model
    model.evaluate(test_loader)

    # Test prediction
    sample_input = torch.tensor(X_test[:5], dtype=torch.float32)
    predictions = model.predict(sample_input)
    print("Predictions:", predictions)