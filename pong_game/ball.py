import pygame
import random
import settings as set

pygame.init()

class Ball:
	def __init__(self, x, y, vx_range, vy_range, size):
		self.vx = 0
		self.vy = 0
		self.vx_range = vx_range
		self.vy_range = vy_range
		self.size = size
		self.rect = pygame.Rect(x, y, size, size)
		self.color = pygame.Color("red")
		self.direction = random.choice(["left","right"])
		self.update(True)

	def update(self, hit):
		# horizontal handling
		if hit:
			if self.direction == "right":
				self.vx = random.randint(self.vx_range[0], self.vx_range[1])
			else:
				self.vx = -random.randint(self.vx_range[0], self.vx_range[1])

		# vertical handling
		if self.rect.y >= set.HEIGHT - self.size:
			if hit:
				self.vy = -random.randint(self.vy_range[0], self.vy_range[1])
			else:
				self.vy = -self.vy
		elif self.rect.y <= 0 + self.size:
			if hit:
				self.vy = random.randint(self.vy_range[0], self.vy_range[1])
			else:
				self.vy = -self.vy

		# wall bounce handling
		self.rect.x += self.vx
		self.rect.y += self.vy
