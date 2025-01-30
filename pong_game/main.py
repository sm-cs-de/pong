# https://github.com/x4nth055/pythoncode-tutorials/tree/master/gui-programming/pong-game

import pygame, sys
from pong import Pong


class Pong:
	def __init__(self, screen, num):
		self.screen = screen
		self.num = num
		self.FPS = pygame.time.Clock()

	def draw(self):
		pygame.display.flip()

	def main(self):
		# start menu here
		game = Pong(self.screen, self.num)  # pass to game the player_option saved to game.game_mode

		while True:
			if self.screen is not None:
				self.screen.fill("black")

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						pygame.quit()
						sys.exit()

				game.player_move()

			game.update()

			if self.screen is not None:
				self.draw()
				self.FPS.tick(30)

		return game.get_data()