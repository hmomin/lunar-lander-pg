<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/lunar_lander_title.png" width="100%" alt="lunar-lander-logo">
</p>

# Introduction

This script trains an agent with stochastic policy gradient ascent to solve the Lunar Lander challenge from OpenAI.

In order to run this script, [NumPy](https://numpy.org/install/), the [OpenAI Gym toolkit](https://gym.openai.com/docs/), and [PyTorch](https://pytorch.org/get-started/locally/) will need to be installed.

Each step through the Lunar Lander environment takes the general form:

```python
state, reward, done, info = env.step(action)
```

and the goal is for the agent to take actions that maximize the cumulative reward achieved for the episode's duration. In this specific environment, the state space is 8-dimensional and continuous, while the action space consists of four discrete options:

- do nothing,
- fire the left orientation engine,
- fire the main engine,
- and fire the right orientation engine.

In order to "solve" the environment, the agent needs to complete the episode with at least 200 points. To learn more about how the agent receives rewards, see [here](https://gym.openai.com/envs/LunarLander-v2/).

# Algorithm

Since the agent can only take one of four actions, <b>a</b>, at each time step <b>t</b>, a natural choice of policy would yield probabilities of each action as its output, given an input state, <b>s</b>. Namely, the policy, <b>π<sub>θ</sub>(a|s)</b>, chosen for the agent is a neural network function approximator, designed to more closely approximate the optimal policy <b>π\*(a|s)</b> of the agent as it trains over more and more episodes. Here, <b>θ</b> represents the parameters of the neural network that are initially randomized but improve over time to produce more optimal actions, meaning those actions that lead to more cumulative reward over time. Each hidden layer of the neural network uses a ReLU activation. The last layer is a softmax layer of four neurons, meaning each neuron outputs the probability that its corresponding action will be selected.

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/lunar_lander_nn.png" width="80%" alt="neural-network">
</p>

Now that the agent has a stochastic mechanism to select output actions given an input state, it begs the question as to how the policy itself improves over episodes. At the end of each episode, the reward, <b>G<sub>t</sub></b>, due to selecting a specific action, <b>a<sub>t</sub></b>, at time <b>t</b> during the episode can be expressed as follows:

<p align="center"><b>G<sub>t</sub> = r<sub>t</sub> + (γ)r<sub>t+1</sub> + (γ<sup>2</sup>)r<sub>t+2</sub> + ...</b></p>

where <b>r<sub>t</sub></b> is the immediate reward and all remaining terms form the discounted sum of future rewards with discount factor <b>0 < γ < 1</b>.

Then, the goal is to change the parameters to increase the expectation of future rewards. By taking advantage of likelihood ratios, a gradient estimator of the form below can be used:

<p align="center"><b>grad = E<sub>t</sub> [ ∇<sub>θ</sub> log( π<sub>θ</sub>( a<sub>t</sub> | s<sub>t</sub> ) ) G<sub>t</sub> ]</b></p>

where the advantage function is given by the total reward <b>G<sub>t</sub></b> produced by the action <b>a<sub>t</sub></b>. Updating the parameters in the direction of the gradient has the net effect of increasing the likelihood of taking actions that were eventually rewarded and decreasing the likelihood of taking actions that were eventually penalized. This is possible because <b>G<sub>t</sub></b> takes into account all the future rewards received as well as the immediate reward.

# Results

Solving the Lunar Lander challenge requires safely landing the spacecraft between two flag posts while consuming limited fuel. The agent's ability to do this was quite abysmal in the beginning.

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/lunar_lander_crash.gif" width="80%" alt="failure...">'
</p>

After training the agent overnight on a GPU, it could gracefully complete the challenge with ease!

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/lunar_lander_success.gif" width="80%" alt="success!">
</p>

Below, the performance of the agent over 214,000 episodes is documented. The light-blue line indicates individual episodic performance, and the black line is a 100-period moving average of performance. The red line marks the 200 point success threshold.

<p align="center">
    <img src="https://dr3ngl797z54v.cloudfront.net/lunar_lander_results.png" width="80%" alt="training-results">
</p>

It took a little over 17,000 episodes before the agent completed the challenge with a total reward of at least 200 points. After around 25,000 episodes, its average performance began to stabilize, yet, it should be noted that there remained a high amount of variance between individual episodes. In particular, even within the last 15,000 episodes of training, the agent failed roughly 5% of the time. Although the agent could easily conquer the challenge, it occasionally could not prevent making decisions that would eventually lead to disastrous consequences.

# Discussion

One caveat with this specific implementation is that it only works with a discrete action space. However, it is possible to adapt the same algorithm to work with a continuous action space. In order to do so, the softmax output layer would have to transform into a sigmoid or tanh layer, nulling the idea that the output layer corresponds to probabilities. Each output neuron would now correspond to the mean, μ, of the (assumed) Gaussian distribution to which each action belongs. In essence, the distributional means themselves would be functions of the input state.

The training process would then consist of updating parameters such that the means shift to favor actions that result in eventual rewards and disfavor actions that are eventually penalized. While it is possible to adapt the algorithm to support continuous action spaces, it has been noted to have relatively poor or limited performance in practice. In actual scenarios involving continuous action spaces, it would almost certainly be preferable to use DDPG, PPO, or a similar algorithm.

# References

- [Introduction to Reinforcement Learning - David Silver](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- [Deep Reinforcement Learning: Pong from Pixels - Andrej Karpathy](https://karpathy.github.io/2016/05/31/rl/)
- [The Likelihood-Ratio Gradient - Tim Vieira](https://timvieira.github.io/blog/post/2019/04/20/the-likelihood-ratio-gradient/)
- [Policy Gradient Loss with Continuous Action Spaces - Stack Overflow Post](https://ai.stackexchange.com/questions/23847/what-is-the-loss-for-policy-gradients-with-continuous-actions)
- [Continuous Control With Deep Reinforcement Learning - Timothy P. Lillicrap et al.](https://arxiv.org/abs/1509.02971)
- [Proximal Policy Optimization Algorithms - John Schulman et al.](https://arxiv.org/abs/1707.06347)

# License

All files in the repository are under the MIT license.
