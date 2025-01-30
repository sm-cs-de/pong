# https://karpathy.github.io/2016/05/31/rl/
import sys
import pygame
import torch
import ann

sys.path.insert(0, './pong_game')
import pong_game.pong as pong
import pong_game.settings as set


if __name__ == "__main__":
    train_pong = pong.Pong(None, 20000)
    train_data = train_pong.main()
    del train_pong

    model = ann.NeuralNetwork(input_size=4, output_size=set.HEIGHT, hidden_layers=[200, 200], learning_rate=0.01)
    model.summary()
    model.train_model(train_data, epochs=50)

    test_pong = pong.Pong(None, 100)
    test_data = test_pong.main()
    del test_pong

    model.evaluate(test_data)

    for batch_index, (batch, label) in enumerate(test_data):
        predictions = model.predict(batch[0])
        print("Target: " + str(torch.argmax(label[0])))
        print("Prediction: " + str(torch.topk(predictions, k=5)) + "\n")

    pygame.init()
    screen = pygame.display.set_mode((set.WIDTH, set.HEIGHT))
    pygame.display.set_caption("Ping Pong")

    play_pong = pong.Pong(screen, 10, model)
    play_pong.main()
