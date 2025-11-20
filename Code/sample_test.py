import torch as T
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

# Example values
mu = T.tensor([0.5, 0.2])  # Example mean for steering and throttle
sigma = T.tensor([0.1, 0.05])  # Example std deviation for steering and throttle

# Create a Normal distribution
probabilities = Normal(mu, sigma)

# Sample and RSample actions
actions_sample = probabilities.sample()
actions_rsample = probabilities.rsample()

# Apply tanh for steering and sigmoid for throttle
steering_sample = T.tanh(actions_sample[0])
throttle_sample = T.sigmoid(actions_sample[1])

steering_rsample = T.tanh(actions_rsample[0])
throttle_rsample = T.sigmoid(actions_rsample[1])

print("Sampled action (steering, throttle):", (steering_sample.item(), throttle_sample.item()))
print("Reparameterized action (steering, throttle):", (steering_rsample.item(), throttle_rsample.item()))

# Visualizing the distributions and actions
x = T.linspace(-3, 3, 100)
steering_dist = Normal(mu[0], sigma[0])
throttle_dist = Normal(mu[1], sigma[1])

steering_probs = T.exp(steering_dist.log_prob(x))
throttle_probs = T.exp(throttle_dist.log_prob(x))

plt.figure(figsize=(12, 6))

# Plot steering distribution
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), steering_probs.numpy(), label='Steering Distribution')
plt.axvline(steering_sample.item(), color='r', linestyle='--', label='Sampled Steering')
plt.axvline(steering_rsample.item(), color='g', linestyle='--', label='Reparameterized Steering')
plt.title('Steering Distribution')
plt.xlabel('Steering Value')
plt.ylabel('Probability')
plt.legend()

# Plot throttle distribution
plt.subplot(1, 2, 2)
plt.plot(x.numpy(), throttle_probs.numpy(), label='Throttle Distribution')
plt.axvline(throttle_sample.item(), color='r', linestyle='--', label='Sampled Throttle')
plt.axvline(throttle_rsample.item(), color='g', linestyle='--', label='Reparameterized Throttle')
plt.title('Throttle Distribution')
plt.xlabel('Throttle Value')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()