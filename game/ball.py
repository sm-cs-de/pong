import pygame
import random
import numpy as np
import config as cfg

pygame.init()

class Ball:
	def __init__(self, x, y, vx_range, vy_range, size):
		self.vx = -random.randint(vx_range[0], vx_range[1])		# in the beginning the ball flies to the AI with random x speed
		self.vy = random.choice([-1,1]) * random.randint(vy_range[0], vy_range[1]) # .. and with random y speed
		self.vx_range = vx_range
		self.vy_range = vy_range
		self.size = size
		self.rect = pygame.Rect(x, y, size, size)
		self.color = pygame.Color("blue")
		self.direction = random.choice(["left", "right"])

	def update(self, hit):
		if hit:
			if self.direction == "right":
				self.vx = random.randint(self.vx_range[0], self.vx_range[1])
			else:
				self.vx = -random.randint(self.vx_range[0], self.vx_range[1])
			self.vy = np.sign(self.vy) * random.randint(self.vy_range[0], self.vy_range[1])

		if self.rect.y >= cfg.HEIGHT - self.size:
			self.vy = -random.randint(self.vy_range[0], self.vy_range[1])
		elif self.rect.y <= 0 + self.size:
			self.vy = random.randint(self.vy_range[0], self.vy_range[1])

		self.rect.x += self.vx
		self.rect.y += self.vy
