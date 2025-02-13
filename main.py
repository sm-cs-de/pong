# https://karpathy.github.io/2016/05/31/rl/
# https://thepythoncode.com/article/build-a-pong-game-in-python
import sys
import pygame
import ann

sys.path.insert(0, 'game')
import game.pong as pong
import game.config as cfg


if __name__ == "__main__":
    train_pong = pong.Pong(None, 10000)
    train_data = train_pong.main()
    del train_pong

    ann = ann.ANN(input_size=4, output_size=cfg.HEIGHT, hidden_layers=[200, 200, 200], learning_rate=0.02)
    ann.summary()
    ann.train_model(train_data, epochs=20)

    test_pong = pong.Pong(None, 500)
    test_data = test_pong.main()
    del test_pong

    ann.evaluate(test_data)

    pygame.init()
    screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
    pygame.display.set_caption("Pong game")

    play_pong = pong.Pong(screen, 10, ann)
    play_pong.main()
