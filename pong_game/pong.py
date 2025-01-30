import random
import pygame
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '..')
import settings as set
import player
import ball
import ann

class Pong:
	def __init__(self, screen, num, model = None):
		self.screen = screen
		self.game_over = False
		self.count = 0
		self.score_limit = num
		self.winner = None
		self.players = self.screen is not None
		self.playerA = None
		self.playerB = None
		self.ball = None
		self.model = model
		self.pred = [[0,0], [0, set.HEIGHT//2], [0,set.HEIGHT]]
		self.data_beg = np.zeros(shape=(num,4), dtype=float)
		self.data_end = np.zeros(shape=(num,set.HEIGHT), dtype=float)
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
			self.playerA = player.Player(0, set.HEIGHT // 2 - (set.player_height // 2), set.player_width, set.player_height)
			self.playerB = player.Player(set.WIDTH - set.player_width,  set.HEIGHT // 2 - (set.player_height // 2), set.player_width, set.player_height)
		self.ball = ball.Ball(set.WIDTH // 2 - set.player_width, set.HEIGHT // 2 - set.player_width, set.ball_vx_range, set.ball_vy_range, set.ball_size)
		self.data_beg[self.count, 0] = self.ball.rect.x / set.WIDTH
		self.data_beg[self.count, 1] = self.ball.rect.y / set.HEIGHT
		self.data_beg[self.count, 2] = self.ball.vx / max(set.ball_vx_range+set.ball_vy_range)
		self.data_beg[self.count, 3] = self.ball.vy / max(set.ball_vx_range+set.ball_vy_range)

	def ball_hit(self):
		hit = False
		if self.ball.rect.left >= set.WIDTH:
			if self.players:
				self.playerA.score += 1
				self.ball.rect.x = set.WIDTH // 2
				time.sleep(1)
			else:
				raise ValueError('not allowed')	# for data generation the ball always flies left so this cant happen
			hit = True

		elif self.ball.rect.right <= 0:
			if self.players:
				self.data_end[self.count, self.ball.rect.y:(self.ball.rect.y+self.ball.rect.height)] = 1 # if ball hit the AI side we want this position
				self.count += 1
				self.playerB.score += 1
				self.ball.rect.x = set.WIDTH // 2
				time.sleep(1)
			else:
				self.data_end[self.count, self.ball.rect.y:(self.ball.rect.y+self.ball.rect.height)] = 1 # if ball hit the AI side we want this position
				self.count += 1
				self.ball.rect.x = random.randint(self.ball.rect.width, set.WIDTH-2*self.ball.rect.width)
				self.ball.rect.y = random.randint(0, set.HEIGHT-self.ball.rect.height)
				self.ball.direction = "left" # for data generation the ball always flies left
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
			pygame.quit()
			if self.players:
				print(f"{self.winner} wins!!")
				sys.exit()
			return True
		else:
			return False

	def update(self):
		end = False

		hit = self.ball_hit()

		if self.players:
			self.show_score()
			self.playerA.update(self.screen)
			self.playerB.update(self.screen)

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

		if hit and not end:
			self.data_beg[self.count, 0] = self.ball.rect.x / set.WIDTH
			self.data_beg[self.count, 1] = self.ball.rect.y / set.HEIGHT
			self.data_beg[self.count, 2] = self.ball.vx / max(set.ball_vx_range+set.ball_vy_range)
			self.data_beg[self.count, 3] = self.ball.vy / max(set.ball_vx_range+set.ball_vy_range)

		if self.model is not None:
			if self.ball.direction == "left":
				if hit:
					self.pred = [[0,0]]
					data_beg = torch.tensor(self.data_beg[self.count], dtype=torch.float)
					pred = self.model.predict(data_beg)
					# softmax = nn.Softmax(dim=0)
					# self.pred = [[1000.0 * softmax(pred)[i].item(), i] for i in  range(set.HEIGHT)]
					self.pred += [[10*pred[i].item() if pred[i].item() >= 0 else 0, i] for i in range(set.HEIGHT)]
					self.pred += [[0, set.HEIGHT]]
			else:
				self.pred = [[0,0], [0, set.HEIGHT//2], [0,set.HEIGHT]]

		if self.players:
			if self.model is not None:
				pygame.draw.polygon(self.screen, pygame.Color('darkorange'), self.pred)
			pygame.draw.rect(self.screen, self.ball.color, self.ball.rect)

		return end

	def main(self):
		end = False
		while not end:
			if self.players:
				self.screen.fill("black")

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						self.winner = "Nobody"
						self.game_end()
				self.player_move()

			end = self.update()

			if self.players:
				self.draw()
				self.FPS.tick(30)

		# for i in range(self.score_limit):
		# 	print(self.data_beg[i])
		# 	print(np.argmax(self.data_end[i]))

		return ann.TrainingData.get_dataloader(self.data_beg, self.data_end, batch_size=20)
