import pygame
import config as cfg


class Player:
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.rect = pygame.Rect(self.x, self.y, width, height)
		self.color = pygame.Color("black")
		self.score = 0

	def move(self, dist):
		self.rect.y = max(0, min(cfg.HEIGHT - cfg.player_height, self.rect.y + dist))

	def update(self, screen):
		pygame.draw.rect(screen, self.color, self.rect)
