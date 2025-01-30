import random

import pygame, time
import sys
from player import Player
from ball import Ball
import settings as set


class Pong:
	def __init__(self, screen, num):
		self.screen = screen
		self.game_over = False
		self.count = 0
		self.score_limit = num
		self.winner = None
		self.players = self.screen is not None
		self.playerA = None
		self.playerB = None
		self.ball = None
		self.data = []
		self.data_all = []
		if self.players:
			self.players = True
			self.font = pygame.font.SysFont('Bauhaus 93', 60)
			self.inst_font = pygame.font.SysFont('Bauhaus 93', 30)
			self.color = pygame.Color("white")
		self.generate_world()
		self.FPS = pygame.time.Clock()

	def draw(self):
		pygame.display.flip()

	# create and add player to the screen
	def generate_world(self):
		if self.players:
			self.playerA = Player(0, set.HEIGHT // 2 - (set.player_height // 2), set.player_width, set.player_height)
			self.playerB = Player(set.WIDTH - set.player_width,  set.HEIGHT // 2 - (set.player_height // 2), set.player_width, set.player_height)
		self.ball = Ball(set.WIDTH // 2 - set.player_width, set.HEIGHT - set.player_width, set.ball_vx_range, set.ball_vy_range, set.ball_size)
		self.data = [self.ball.rect.x, self.ball.rect.y, self.ball.vx, self.ball.vy, 0.0]

	def ball_hit(self):
		# if ball is not hit and pass through table sides
		hit = False
		if self.ball.rect.left >= set.WIDTH:
			self.data[4] = set.HEIGHT // 2 # if ball traveled to player side we want the middle of the area as target
			if self.players:
				self.playerA.score += 1
				self.ball.rect.x = set.WIDTH // 2
				time.sleep(1)
			else:
				self.count += 1
				self.ball.rect.x = random.randint(self.ball.rect.width, set.WIDTH-2*self.ball.rect.width)
				self.ball.rect.y = random.randint(0, set.HEIGHT-self.ball.rect.height)
				self.ball.direction = random.choice(["left", "right"]) # for data generation we use random directions
			hit = True

		elif self.ball.rect.right <= 0:
			self.data[4] = self.ball.rect.y # if ball hit the AI side we want this position
			if self.players:
				self.playerB.score += 1
				self.ball.rect.x = set.WIDTH // 2
				time.sleep(1)
			else:
				self.count += 1
				self.ball.rect.x = random.randint(self.ball.rect.width, set.WIDTH-2*self.ball.rect.width)
				self.ball.rect.y = random.randint(0, set.HEIGHT-self.ball.rect.height)
				self.ball.direction = random.choice(["left", "right"]) # for data generation we use random directions
			hit = True

		if self.players:
			if pygame.Rect.colliderect(self.ball.rect, self.playerA.rect):
				self.ball.direction = "right"
				hit = True
			if pygame.Rect.colliderect(self.ball.rect, self.playerB.rect):
				self.ball.direction = "left"
				hit = True

		return hit

	def bot_opponent(self):
		if self.ball.direction == "left" and self.ball.rect.centery != self.playerA.rect.centery:
			if self.ball.rect.top <= self.playerA.rect.top:
				if self.playerA.rect.top > 0:
					self.playerA.move_up()
			if self.ball.rect.bottom >= self.playerA.rect.bottom:
				if self.playerA.rect.bottom < set.HEIGHT:
					self.playerA.move_bottom()

	def player_move(self):
		keys = pygame.key.get_pressed()
		self.bot_opponent()

		if keys[pygame.K_UP]:
			if self.playerB.rect.top > 0:
				self.playerB.move_up()
		if keys[pygame.K_DOWN]:
			if self.playerB.rect.bottom < set.HEIGHT:
				self.playerB.move_bottom()

	def show_score(self):
		scoreA = self.font.render(str(self.playerA.score), True, self.color)
		scoreB = self.font.render(str(self.playerB.score), True, self.color)
		self.screen.blit(scoreA, (set.WIDTH // 4, 50))
		self.screen.blit(scoreB, ((set.WIDTH // 4) * 3, 50))

	def game_end(self):
		if self.winner is not None:
			if self.players:
				print(f"{self.winner} wins!!")
			pygame.quit()
			return True
		else:
			return False

	def update(self):
		end = False

		if self.players:
			self.show_score()

			self.playerA.update(self.screen)
			self.playerB.update(self.screen)

		hit = self.ball_hit()

		if self.players:
			if self.playerA.score == self.score_limit:
				self.winner = "Opponent"
			elif self.playerB.score == self.score_limit:
				self.winner = "You"
			end = self.game_end()

		else:
			if self.count >= self.score_limit:
				self.winner = ""
			end = self.game_end()

		self.ball.update(hit)
		if self.players:
			pygame.draw.rect(self.screen, self.ball.color, self.ball.rect)

		if hit:
			self.data_all.append(self.data)
			print(self.data)
			self.data[0] = self.ball.rect.x
			self.data[1] = self.ball.rect.y
			self.data[2] = self.ball.vx
			self.data[3] = self.ball.vy

		return end

	def main(self):
		end = False
		while not end:
			if self.players:
				self.screen.fill("black")

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						pygame.quit()
						sys.exit()
				self.player_move()

			end = self.update()

			if self.players:
				self.draw()
				self.FPS.tick(30)

		return self.data
