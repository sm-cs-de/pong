# https://karpathy.github.io/2016/05/31/rl/
import sys
import pygame
import torch
import ann
import dqn

sys.path.insert(0, './pong_game')
import pong_game.pong as pong
import pong_game.config as cfg


if __name__ == "__main__":
    train_pong = pong.Pong(None, 20000)
    train_data = train_pong.main()
    del train_pong

    ann = ann.ANN(input_size=4, output_size=cfg.HEIGHT, hidden_layers=[200, 200, 200], learning_rate=0.01)
    ann.summary()
    ann.train_model(train_data, epochs=20)

    test_pong = pong.Pong(None, 50)
    test_data = test_pong.main()
    del test_pong

    ann.evaluate(test_data)

    dqn = dqn.DQN()

    pygame.init()
    screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
    pygame.display.set_caption("Ping Pong")

    play_pong = pong.Pong(screen, 10, ann, dqn)
    play_pong.main()
