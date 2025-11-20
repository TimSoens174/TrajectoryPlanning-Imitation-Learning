import torch as T
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
from SACNetwork import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[12],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.0005, decay_episodes=100):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_episodes = decay_episodes
        
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

        self.mu_list = []
        self.sigma_list = []

    def choose_action(self, observation, episode_num):
        # If still within epsilon-greedy exploration phase
        if np.random.random() < self.epsilon:
            # Epsilon-greedy random actions:
            steering = np.random.uniform(-1, 1)  # Random steering between -1 and 1
            throttle = np.random.uniform(0, 1)   # Random throttle between 0 and 1
            action = np.array([steering, throttle])
            print("random action: ", action)
            self.update_epsilon()
        else:
            # Use policy (Gaussian) to choose action
            state = T.Tensor([observation]).to(self.actor.device)
            action, (mu, sigma), _ = self.actor.sample_normal(state, reparameterize=False)
            action = action.cpu().detach().numpy()[0]  # Convert tensor to numpy array

            # Separate steering and throttle if necessary, ensuring correct ranges
            steering = np.clip(action[0], -1, 1)  # Steering within range [-1, 1]
            throttle = np.clip(action[1], 0, 1)   # Throttle within range [0, 1]
            action = np.array([steering, throttle])
            print("network action: ", action)

            # Store mu and sigma for tracking
            self.mu_list.append(mu.cpu().detach().numpy())
            self.sigma_list.append(sigma.cpu().detach().numpy())

        return action

    def update_epsilon(self):
        """ Decay epsilon after every episode. """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def print_mean_mu_sigma(self):
        """ Calculate and print mean of mu and sigma after an episode """
        if self.mu_list and self.sigma_list:
            mean_mu = np.mean(self.mu_list, axis=0)
            mean_sigma = np.mean(self.sigma_list, axis=0)

            print(f"Mean mu for the episode: {mean_mu}")
            print(f"Mean sigma for the episode: {mean_sigma}")
        
        # Clear lists for the next episode
        self.mu_list = []
        self.sigma_list = []

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, _,log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, _, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        value_dict = dict(value_params)

        for name, param in target_value_params:
            value_dict[name] = tau * value_dict[name].clone() + \
                               (1 - tau) * param.clone()

        self.target_value.load_state_dict(value_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
