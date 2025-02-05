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
	def __init__(self, screen, num, ann = None, dqn = None):
		self.screen = screen
		self.game_over = False
		self.count = 0
		self.score_limit = num
		self.winner = None
		self.players = self.screen is not None
		self.player_cpu = None
		self.player_hum = None
		self.ball = None
		self.ann = ann
		self.dqn = dqn
		self.ball_dist = None
		self.ball_prob = None
		self.ball_goal = None
		self.ball_start = np.zeros(shape=(num, 4), dtype=float)
		self.ball_end = np.zeros(shape=(num, cfg.HEIGHT), dtype=float)
		if self.players:
			self.players = True
			self.font = pygame.font.SysFont('Bauhaus 93', 60)
			self.inst_font = pygame.font.SysFont('Bauhaus 93', 30)
			self.color = pygame.Color("white")
		self.generate()
		self.FPS = pygame.time.Clock()

	def draw(self):
		pygame.display.flip()

	def generate(self):
		if self.players:
			self.player_cpu = player.Player(0, cfg.HEIGHT // 2 - (cfg.player_height // 2), cfg.player_width, cfg.player_height)
			self.player_hum = player.Player(cfg.WIDTH - cfg.player_width, cfg.HEIGHT // 2 - (cfg.player_height // 2), cfg.player_width, cfg.player_height)
		self.ball = ball.Ball(cfg.WIDTH // 2 - cfg.player_width, cfg.HEIGHT // 2 - cfg.player_width, cfg.ball_vx_range, cfg.ball_vy_range, cfg.ball_size)
		self.ball_start[self.count, 0] = self.ball.rect.x / cfg.WIDTH
		self.ball_start[self.count, 1] = self.ball.rect.y / cfg.HEIGHT
		self.ball_start[self.count, 2] = self.ball.vx / max(cfg.ball_vx_range + cfg.ball_vy_range)
		self.ball_start[self.count, 3] = self.ball.vy / max(cfg.ball_vx_range + cfg.ball_vy_range)

		if self.ann is not None:
			self.ball_dist = torch.tensor(np.zeros(cfg.HEIGHT), dtype=torch.float)
			self.ball_dist[cfg.HEIGHT//2 - cfg.ball_size//2 : cfg.HEIGHT//2 + cfg.ball_size//2] = 1.0 / cfg.ball_size  # we use the mid as target
			# if self.dqn is not None:
			# 	state = np.concatenate([self.ball_dist, [self.player_cpu.y]], axis=0)
			# 	self.dqn.buffer.push(state, cfg.player_move_distances.index(0), 1, state, False)

	def get_ball_prob(self, ann_output):
		conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=cfg.player_width, padding='same', padding_mode='zeros', bias=False)
		conv.weight.data = torch.full_like(conv.weight.data, 1)

		return conv(ann_output.unsqueeze(0))[0]

	def get_ball_goal(self, ann_output):
		return np.sum([i * ann_output[i].item() for i in range(ann_output.size(dim=0))]) / torch.sum(ann_output).item()  # we calculate the expectation value (y-index) of the output (distribution)

	def get_ball_hit(self):
		hit = False
		if self.ball.rect.left >= cfg.WIDTH:
			if self.players:
				self.player_cpu.score += 1
				self.ball.rect.x = cfg.WIDTH // 2
				time.sleep(1)
			hit = True

		elif self.ball.rect.right <= 0:
			self.ball_end[self.count, self.ball.rect.y : (self.ball.rect.y + cfg.ball_size)] = 1.0  # only relevant for training; we use the whole y impact location and larger values to improve training process
			self.count += 1
			if self.players:
				self.player_hum.score += 1
				self.ball.rect.x = cfg.WIDTH // 2
				time.sleep(1)
			else:
				self.ball.rect.x = random.randint(0, cfg.WIDTH - self.ball.rect.width)
				self.ball.rect.y = random.randint(0, cfg.HEIGHT - self.ball.rect.height)
				self.ball.direction = "left" # for data generation the ball always flies left
			hit = True

		if self.players:
			if pygame.Rect.colliderect(self.ball.rect, self.player_cpu.rect):
				self.ball.direction = "right"
				hit = True
			if pygame.Rect.colliderect(self.ball.rect, self.player_hum.rect):
				self.ball.direction = "left"
				hit = True

		return hit

	def player_cpu_move_default(self):
		if self.ball.direction == "left" and self.ball.rect.centery != self.player_cpu.rect.centery:
			if self.ball.rect.top <= self.player_cpu.rect.top:
				self.player_cpu.move(cfg.player_move_distances[0])
			if self.ball.rect.bottom >= self.player_cpu.rect.bottom:
				self.player_cpu.move(cfg.player_move_distances[-1])

	def player_cpu_move_agent(self, distance):
		self.player_cpu.move(distance)

	def player_hum_move(self):
		keys = pygame.key.get_pressed()
		if keys[pygame.K_UP]:
			self.player_hum.move(cfg.player_move_distances[0])
		if keys[pygame.K_DOWN]:
			self.player_hum.move(cfg.player_move_distances[-1])

	def show_score(self):
		scoreA = self.font.render(str(self.player_cpu.score), True, self.color)
		scoreB = self.font.render(str(self.player_hum.score), True, self.color)
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

		hit = self.get_ball_hit()

		if self.players:
			if self.player_cpu.score == self.score_limit:
				self.winner = "Opponent"
			elif self.player_hum.score == self.score_limit:
				self.winner = "You"
			end = self.game_end()
		else:
			if self.count >= self.score_limit:
				self.winner = ""
			end = self.game_end()

		self.ball.update(hit)

		if not end:
			if hit or self.players:
				self.ball_start[self.count, 0] = self.ball.rect.x / cfg.WIDTH
				self.ball_start[self.count, 1] = self.ball.rect.y / cfg.HEIGHT
				self.ball_start[self.count, 2] = self.ball.vx / max(cfg.ball_vx_range + cfg.ball_vy_range)
				self.ball_start[self.count, 3] = self.ball.vy / max(cfg.ball_vx_range + cfg.ball_vy_range)

		cpu_move_distance = 0
		if self.ann is not None:
			if self.ball.direction == "left":
				data_beg = torch.tensor(np.array([self.ball_start[self.count]]), dtype=torch.float)
				self.ball_dist = self.ann.predict(data_beg)
			else:
				self.ball_dist = torch.tensor(np.zeros(cfg.HEIGHT), dtype=torch.float)
				self.ball_dist[cfg.HEIGHT // 2 - cfg.ball_size // 2: cfg.HEIGHT // 2 + cfg.ball_size // 2] = 1.0 / cfg.ball_size # we use the mid as target
			self.ball_prob = self.get_ball_prob(self.ball_dist)
			self.ball_goal = self.get_ball_goal(self.ball_dist)

			if self.dqn is not None:
				player_cpu_y = self.player_cpu.rect.y
				print("## DQGN for " + str(player_cpu_y))

				self.dqn.reset()
				state = np.concatenate([self.ball_prob.tolist(), [player_cpu_y]], axis=0)
				self.dqn.buffer.push(state, cfg.player_move_distances.index(0), 1, state, False)

				for i in range(0, 100):
					cpu_move_action = self.dqn.get_action(state)
					player_cpu_y = max(0, min(cfg.HEIGHT - cfg.player_height, player_cpu_y + cfg.player_move_distances[cpu_move_action]))
					# self.player_cpu_move_agent(cpu_move_distance)
					reward = self.ball_prob[player_cpu_y].item()
					done = False
					state[-1] = player_cpu_y
					self.dqn.buffer.append(state, cpu_move_action, reward, done)
					self.dqn.train_step()
					print(f"{i}: {cpu_move_action} {player_cpu_y} {reward} {state[-1]}")

				cpu_move_action = self.dqn.get_action(state)
				cpu_move_distance = cfg.player_move_distances[cpu_move_action]
				print(cpu_move_distance)

		if self.players:
			if self.ann is not None:
				ball_prob = [[0, 0]] + [[100 / torch.max(self.ball_prob).item() * self.ball_prob[i].item(), i] for i in range(cfg.HEIGHT)] + [[0, cfg.HEIGHT]]
				pygame.draw.polygon(self.screen, pygame.Color('darkorange'), ball_prob)

				ball_goal = [[0, self.ball_goal], [100, self.ball_goal]]
				pygame.draw.lines(self.screen, pygame.Color('red'), False, ball_goal, width=3)
			pygame.draw.rect(self.screen, self.ball.color, self.ball.rect)

		if self.players:
			self.show_score()
			self.player_cpu.update(self.screen)
			self.player_hum.update(self.screen)

		return cpu_move_distance, end

	def main(self):
		end = False
		while not end:

			if self.players:
				self.screen.fill("black")

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						self.winner = "Nobody"
						self.game_end()

			cpu_move_distance, end = self.update()

			if self.players:
				self.player_hum_move()
				if self.ann is not None and self.dqn is not None:
					self.player_cpu_move_agent(cpu_move_distance)
				else:
					self.player_cpu_move_default()

				self.draw()
				self.FPS.tick(30)

		return ann.TrainingData.get_dataloader(self.ball_start, self.ball_end)
