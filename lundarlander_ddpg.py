from ddpg import Agent
import gym
import numpy as np
import wandb
wandb.init(project="policygradients")




if __name__ =='__main__':
	env = gym.make('LunarLanderContinuous-v2')
	agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, n_actions=2, layer1_dims=400, layer2_dims=300, batch_size=64)

	np.random.seed(0)
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
			agent.remember(observation, action, reward, observation_, int(done))
			agent.learn()
			observation = observation_
			score += reward
		score_history.append(score)
		wandb.log({"score": score})