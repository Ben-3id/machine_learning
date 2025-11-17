import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from thinker import Qnet, QTrain
from helper import plot


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

memory_len = 400_000

LR=0.00001
Gamma=0.80
BATCH_SIZE = 10000
class Agent:
    def __init__(self):
        self.n_games=0
        self.game=SnakeGameAI()
        self.model=Qnet(input=11,output=3)
        self.trainer=QTrain(self.model,lr=LR,gamma=Gamma)
        self.memory=deque(maxlen=memory_len)

    def get_state(self,game):
        head=game.snake[0]
        point_l=Point(head.x-20,head.y)
        point_r=Point(head.x+20,head.y)
        point_u=Point(head.x,head.y-20)
        point_d=Point(head.x,head.y+20)
        dir_l= game.direction == Direction.LEFT 
        dir_r= game.direction == Direction.RIGHT 
        dir_u= game.direction == Direction.UP   
        dir_d= game.direction == Direction.DOWN 

        point_l2=Point(head.x-40,head.y)
        point_r2=Point(head.x+40,head.y)
        point_u2=Point(head.x,head.y-40)
        point_d2=Point(head.x,head.y+40)
        state=[
            #danger straight 
            (game.is_collision(point_l) and dir_l) or
            (game.is_collision(point_r) and dir_r) or
            (game.is_collision(point_u) and dir_u) or
            (game.is_collision(point_d) and dir_d) ,

            #danger right
            (game.is_collision(point_l) and dir_u) or
            (game.is_collision(point_r) and dir_d) or
            (game.is_collision(point_u) and dir_r) or
            (game.is_collision(point_d) and dir_l) ,

            #danger left
            (game.is_collision(point_l) and dir_d) or
            (game.is_collision(point_d) and dir_r) or
            (game.is_collision(point_r) and dir_u) or
            (game.is_collision(point_u) and dir_l) ,


            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.is_collision(point_l2),
            game.is_collision(point_r2),
            game.is_collision(point_u2),
            game.is_collision(point_d2),


            game.food.x > head.x,
            game.food.x < head.x,
            game.food.y > head.y,
            game.food.y < head.y,


        ]
        return np.array(state, dtype=int)

    def get_action(self,state):
        move=[0,0,0]
        epsilon = 80 - self.n_games

        if random.randint(0,200) < epsilon:
            idx=random.randint(0,2)
            move[idx]=1
        else:
            stateT=torch.tensor(state,dtype=torch.float)
            pred=self.model(stateT)
            idx=torch.argmax(pred).item()
            move[idx]=1
        return move

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))


    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train(state,action,reward,next_state,done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train(states, actions, rewards, next_states, dones)    
    
def play():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True :
        old_states=agent.get_state(game)

        move=agent.get_action(old_states)

        reward, game_over, score =game.play_step(move)

        next_state=agent.get_state(game)

        agent.remember(old_states,move,reward,next_state,game_over)

        agent.train_short_memory(old_states,move,reward,next_state,game_over)

        
        if game_over:
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()

            if record < score :
                record = score
                agent.model.save()

            print(f"game : {agent.n_games} -----> score : {score}  ---best one---> {record}")

            total_score+=score
            plot_mean_scores.append(total_score/agent.n_games)
            plot_scores.append(score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    play()