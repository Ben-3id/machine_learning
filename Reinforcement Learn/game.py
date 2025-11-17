import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# --- ENUMS & STRUCTURES ---
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# --- COLORS ---
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 70, 70)
BLUE_HEAD = (50, 150, 255)
BLUE_BODY = (30, 90, 200)
GRID_COLOR = (40, 40, 40)

# --- GAME SETTINGS ---
BLOCK_SIZE = 20
SPEED = 200

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class SnakeGameAI:
    
    def __init__(self, w=1000, h=800):
        self.w = w
        self.h = h
        self.reward=0
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.display
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.reward=0
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self.adv_move(action)
        self.snake.insert(0, self.head)
        self.reward -=0.1
        # 3. check if game over
        game_over = False
        if self.is_collision():
            self.reward -= 10
            game_over = True
            return self.reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.reward += 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return self.reward, game_over, self.score
    

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # collision with self
        if pt in self.snake[1:]:
            return True
        # collision with wall (no wrap-around)
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        return False
    
    def adv_move(self,action):
        food=np.array(self.food)
        head=np.array(self.head)
        before=manhattan(food,head)
        self._move(action)
        food=np.array(self.food)
        head=np.array(self.head)
        after=manhattan(food,head)
        if before > after:
            self.reward +=0.2
        else : self.reward -=0.2
        
    def _update_ui(self):
        # Gradient background
        self.display.fill(BLACK)
        for y in range(self.h):
            color = (20 + y // 25, 20 + y // 30, 30 + y // 20)
            pygame.draw.line(self.display, color, (0, y), (self.w, y))

        # Optional grid
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.w, y))

        # Draw snake with rounded corners and fading color
        # Draw snake with gradient body and distinct head color
        head_color = (0, 255, 255)  # bright cyan head
        tail_color = (0, 60, 120)   # deep blue tail

        for i, pt in enumerate(self.snake):
            if i == 0:
                # Head
                color = head_color
            else:
                # Gradient body
                t = i / len(self.snake)
                r = int(head_color[0] * (1 - t) + tail_color[0] * t)
                g = int(head_color[1] * (1 - t) + tail_color[1] * t)
                b = int(head_color[2] * (1 - t) + tail_color[2] * t)
                color = (r, g, b)

            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
                border_radius=6
            )

        # Draw food with pulsing glow
        pulse = int(120 + 50 * math.sin(pygame.time.get_ticks() * 0.005))
        pulse = max(0, min(255, pulse))  # ensure it's within [0, 255]
        glow_color = (pulse, 50, 50)

        pygame.draw.rect(
            self.display,
            glow_color,
            pygame.Rect(self.food.x - 2, self.food.y - 2, BLOCK_SIZE + 4, BLOCK_SIZE + 4),
            border_radius=8
        )
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
            border_radius=6
        )

        # Draw score (with shadow)
        text_surface = font.render(f"Score: {self.score}", True, WHITE)
        shadow = font.render(f"Score: {self.score}", True, (50, 50, 50))
        self.display.blit(shadow, (3, 3))
        self.display.blit(text_surface, (0, 0))

        pygame.display.flip()
        
    def _move(self, action):
        moves = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = moves.index(self.direction)
        
        if np.array_equal(action, [0, 1, 0]):
            self.direction = moves[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            self.direction = moves[(idx - 1) % 4]
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

            
        self.head = Point(x, y)


