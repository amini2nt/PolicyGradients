import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer(object):
	def __init__(self, max_size, input_shape, n_actions):
		self.mem_size = max_size
		self.mem_cntr = 0
		self.state_memory = np.zeros((self.mem_size, *input_shape))
		self.new_state_memory = np.zeros((self.mem_size, *input_shape))
		self.action_memory = np.zeros((self.mem_size, n_actions))
		self.reward_memory = np.zeros(self.mem_size)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.new_state_memory[index] = state_
		self.terminal_memory[index] = 1 - done ##see the bellman equation later on. 
		self.mem_cntr += 1 

	def sample_buffer(self,  batch_size):
		max_mem = min(self.mem_size, self.mem_cntr)
		batch = np.random.choice(max_mem, batch_size)

		states = self.state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		new_states = self.new_state_memory[batch]
		terminals = self.terminal_memory[batch]

		return states, actions, rewards, new_states, terminal 


class Critic(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='temp/ddpg'):
		super(Critic, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.checkpoint_file = os.pth.join(chkpt_dir, name + '_ddpg')

		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		###neeed to initialize the weights and biases in a constrained way --> explained in the paper for faster convergence
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		self.bn1 = nn.LayerNorm(self.fc1_dims)

		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
		self.bn2 = nn.LayerNorm(self.fc2_dims)

		###When you have a normalization layer you gotta make sure you use the .train(), .eval() functions properly

		self.action_value = nn.Linear(self.n_actions, fc2_dims)
		f3 = 0.003
		self.q = nn.Linear(self.fc2_dims, 1)
		torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
		torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self,state, action):
		state_value = self.fc1(state)
		state_value = self.bn1(state_value)
		state_value = F.relu(state_value)

		state_value = self.fc2(state_value)
		state_value = self.bn2(state_value)

		action_value = F.relu(self.action_value(action))
		state_action_value = F.relu(torch.add(state_value, action_value)) ## this double relu twice on action value is a bit sketchy
		state_action_value = self.q(state_action_value)

	def save_checkpoint(self):
		print('.... saving checkpoint .....')
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('.... loading checkpoint .....')
		torch.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='temp/ddpg'):
		super(Actor, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.checkpoint_file = os.pth.join(chkpt_dir, name + '_ddpg')

		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		###neeed to initialize the weights and biases in a constrained way --> explained in the paper for faster convergence
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		self.bn1 = nn.LayerNorm(self.fc1_dims)

		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
		self.bn2 = nn.LayerNorm(self.fc2_dims)

		# ##When you have a normalization layer you gotta make sure you use the .train(), .eval() functions properly

		f3 = 0.003
		self.mu = nn.Linear(self.fc2_dims, self.n_actions)
		torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
		torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self,state):
		selected_action = self.fc1(state)
		selected_action = self.bn1(selected_action)
		selected_action = F.relu(selected_action)

		selected_action = self.fc2(selected_action)
		selected_action = self.bn2(selected_action)
		selected_action = F.relu(selected_action)

		selected_action = torch.tanh(self.mu(selected_action))#bound it by -1 and 1. We can later on bound it by the action upper and lower boudn of the specific environment
		return selected_action 

	def save_checkpoint(self):
		print('.... saving checkpoint .....')
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('.... loading checkpoint .....')
		torch.load_state_dict(torch.load(self.checkpoint_file))



class Agent(object):
	def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, layer1_dims=400, layer2_dims=300, batch_size=64):
	 	self.gamma = gamma
	 	self.tau = tau
	 	self.memory = ReplayBuffer(max_size, input_dims, n_actions)
	 	self.batch_size = batch_size

	 	self.actor = Actor(alpha, input_dims, layer1_dims, layer2_dims, n_actions, name='Actor')
	 	self.target_actor = Actor(alpha, input_dims, layer1_dims, layer2_dims, n_actions, name='target_Actor')

	 	self.critic = Critic(beta, input_dims, layer1_dims, layer2_dims, n_actions, name='Critic')
	 	self.target_critic = Critic(beta, input_dims, layer1_dims, layer2_dims, n_actions, name='target_Critic')

	 	self.noise_dist =  torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.5])) 

	 	self.update_nw_params(tau=1)

	def chooose_action(self, observation):
		self.actor.eval()
		observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
		mu = self.actor(observation).to(self.actor.device)
		mu_prime = mu + torch.tensor(self.noise_dist.sample(sample_shape=mu.shape), dtype=torch.float).to(self.actor.device)
		self.actor.train()
		return mu_prime.cpu().detach().numpy() ## so that we can actually take the action in openai gym   

	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)


	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
		reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
		done = torch.tensor(done).to(self.critic.device)
		new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
		action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
		state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
		
		self.target_actor.eval()
		self.target_critic.eval()
		self.critic.eval()

		target_actions = self.target_actor.forward(new_state)
		critic_value_ = self.target_critic.forward(new_state, target_actions)

		critic_value = self.critic.forward(state, action)

		target = []
		for j in range(self.batch_size):
			target.append(rewards[j] + self.gamma*critic_value_[j]*done[j])
		target = torch.tensor(target).to(self.critic.device)
		target =  target.view(self.batch_size, 1)

		self.critic.train()
		self.critic.optimizer.zero_grad()
		critic_loss = F.mse_loss(target, critic_value)
		critic_loss.backward()
		self.critic.optimizer.step()

		self.critic.eval()
		self.critic.optimizer.zero_grad()
		mu = self.actor.forward(state)
		self.actor.train()
		actor_loss = -self.critic.forward(state, mu)
		actor_loss = torch.mean(torch.actor_loss)
		self.actor.optimizer.step()

		self.update_nw_params()

	def update_nw_params(self, tau=None):
		if tau is None:
			tau = self.tau

		actor_params_dict = dict(self.actor.named_parameters()) ### gets the names of the network
		critic_params_dict = dict(self.critic.named_parameters())
		target_actor_params_dict = dict(self.target_actor.named_parameters())
		target_critic_params_dict = dict(self.target_critic.named_parameters())


		for name in critic_params_dict:
			critic_params_dict[name] = tau*critic_params_dict[name].clone() + (1-tau) * target_critic_params_dict[name].clone()

		self.target_critic.load_state_dict(critic_para,s_dict)


		for name in actor_params_dict:
			actor_params_dict[name] = tau*actor_params_dict[name].clone() + (1-tau) * target_actor_params_dict[name].clone()

		self.target_actor.load_state_dict(actor_params_dict)

	def save_model(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic.save_checkpoint()

	def load_model(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.target_critic.load_checkpoint()






