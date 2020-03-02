import numpy as np
import gym
from policygradient import Agent
import wandb
wandb.init(project="policygradients")

if __name__ =='__main__':
	env = gym.make('LunarLander-v2')
	agent = Agent(lr=0.001, input_dims=[8], gamma=0.99, n_actions = 4, l1_size = 128, l2_size=128)
	score_history = []
	score = 0
	n_episodes = 2500

	for i in range(n_episodes):
		print('episode: ', i,  'score %.3f' % score)
		done = False
		score = 0
		observation = env.reset()
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.store_reward(reward)
			observation = observation_
			score += reward
		score_history.append(score)
		agent.learn()
		wandb.log({"score": score})