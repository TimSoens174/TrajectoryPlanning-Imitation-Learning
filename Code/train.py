import numpy as np
from SACAgent import Agent
from utils import plot_learning_curve
from CarEnv import CarEnv
import generate_traffic

if __name__ == '__main__':
    env = CarEnv(mount_camera=True)
    N_EPISODES = 10000
    MAX_STEP_COUNT = 20
    agent = Agent(input_dims=env.tracking_data.shape, env=env,
            n_actions=2, decay_episodes=N_EPISODES//5)

    filename = 'sac_carla.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    # Spawn traffic
        #generate_traffic.main()

    for i in range(N_EPISODES):
        state = env.reset()

        if state is None:
            print(f"Skipping episode {i} due to vehicle spawn failure.")
            continue  # Skip this episode and move to the next one
        done = False
        score = 0
        step_counter = 0
        while not done and step_counter <= MAX_STEP_COUNT:
            step_counter += 1
            action = agent.choose_action(state, i)
            #print("Action: ", action)
            state_, reward, done, info = env.step(action)
            score += reward
            agent.remember(state, action, reward, state_, done)
            if not load_checkpoint:
                agent.learn()
            state = state_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print("-----------------------------------------------------------------------------------------")
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        # Call to print mean mu and sigma for the episode
        agent.print_mean_mu_sigma()
        # Destroy actors at the end of each episode to avoid leaks
        env.destroy()

    if not load_checkpoint:
        x = [i+1 for i in range(N_EPISODES)]
        plot_learning_curve(x, score_history, figure_file)