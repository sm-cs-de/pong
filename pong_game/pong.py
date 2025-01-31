import random
import pygame
import sys
import time
import numpy as np
import torch

sys.path.insert(0, '..')
import config as cfg
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
		self.model_prob = [[0, 0], [0, cfg.HEIGHT // 2], [0, cfg.HEIGHT]]
		self.model_goal = [[0,0], [0,0]]
		self.data_beg = np.zeros(shape=(num,4), dtype=float)
		self.data_end = np.zeros(shape=(num, cfg.HEIGHT), dtype=float)
		if self.players:
			self.players = True
			self.font = pygame.font.SysFont('Bauhaus 93', 60)
			self.inst_font = pygame.font.SysFont('Bauhaus 93', 30)
			self.color = pygame.Color("white")
		self.generate_world()
		self.FPS = pygame.time.Clock()

	def draw(self):
		pygame.display.flip()

	def generate_world(self):
		if self.players:
			self.playerA = player.Player(0, cfg.HEIGHT // 2 - (cfg.player_height // 2), cfg.player_width, cfg.player_height)
			self.playerB = player.Player(cfg.WIDTH - cfg.player_width, cfg.HEIGHT // 2 - (cfg.player_height // 2), cfg.player_width, cfg.player_height)
		self.ball = ball.Ball(cfg.WIDTH // 2 - cfg.player_width, cfg.HEIGHT // 2 - cfg.player_width, cfg.ball_vx_range, cfg.ball_vy_range, cfg.ball_size)
		self.data_beg[self.count, 0] = self.ball.rect.x / cfg.WIDTH
		self.data_beg[self.count, 1] = self.ball.rect.y / cfg.HEIGHT
		self.data_beg[self.count, 2] = self.ball.vx / max(cfg.ball_vx_range + cfg.ball_vy_range)
		self.data_beg[self.count, 3] = self.ball.vy / max(cfg.ball_vx_range + cfg.ball_vy_range)

	def ball_hit(self):
		hit = False
		if self.ball.rect.left >= cfg.WIDTH:
			if self.players:
				self.playerA.score += 1
				self.ball.rect.x = cfg.WIDTH // 2
				time.sleep(1)
			hit = True

		elif self.ball.rect.right <= 0:
			self.data_end[self.count, self.ball.rect.y:(self.ball.rect.y+self.ball.rect.height)] = 1 # if ball hit the AI side we want this position
			self.count += 1
			if self.players:
				self.playerB.score += 1
				self.ball.rect.x = cfg.WIDTH // 2
				time.sleep(1)
			else:
				self.ball.rect.x = random.randint(0, cfg.WIDTH - self.ball.rect.width)
				self.ball.rect.y = random.randint(0, cfg.HEIGHT - self.ball.rect.height)
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
				if self.playerA.rect.bottom < cfg.HEIGHT:
					self.playerA.move_bottom()

	def player_move(self):
		keys = pygame.key.get_pressed()
		self.bot_opponent()

		if keys[pygame.K_UP]:
			if self.playerB.rect.top > 0:
				self.playerB.move_up()
		if keys[pygame.K_DOWN]:
			if self.playerB.rect.bottom < cfg.HEIGHT:
				self.playerB.move_bottom()

	def show_score(self):
		scoreA = self.font.render(str(self.playerA.score), True, self.color)
		scoreB = self.font.render(str(self.playerB.score), True, self.color)
		self.screen.blit(scoreA, (cfg.WIDTH // 4, 50))
		self.screen.blit(scoreB, ((cfg.WIDTH // 4) * 3, 50))

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

		if not end:
			if hit or self.players:
				self.data_beg[self.count, 0] = self.ball.rect.x / cfg.WIDTH
				self.data_beg[self.count, 1] = self.ball.rect.y / cfg.HEIGHT
				self.data_beg[self.count, 2] = self.ball.vx / max(cfg.ball_vx_range + cfg.ball_vy_range)
				self.data_beg[self.count, 3] = self.ball.vy / max(cfg.ball_vx_range + cfg.ball_vy_range)

		if self.model is not None:
			if self.ball.direction == "left":
				data_beg = torch.tensor(np.array([self.data_beg[self.count]]), dtype=torch.float)
				pred = self.model.predict(data_beg)[0]
				goal = self.model.goal(pred)

				self.model_prob = [[0, 0]]
				self.model_prob += [[100 / torch.max(pred).item() * pred[i].item(), i] for i in range(cfg.HEIGHT)]
				self.model_prob += [[0, cfg.HEIGHT]]
				self.model_goal = [[0, goal], [75, goal]]
			else:
				self.model_prob = [[0, 0], [0, cfg.HEIGHT // 2], [0, cfg.HEIGHT]]
				self.model_goal = [[0,0], [0,0]]

		if self.players:
			if self.model is not None:
				pygame.draw.polygon(self.screen, pygame.Color('darkorange'), self.model_prob)
				pygame.draw.lines(self.screen, pygame.Color('red'), False, self.model_goal, width=3)
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

		return ann.TrainingData.get_dataloader(self.data_beg, self.data_end)
