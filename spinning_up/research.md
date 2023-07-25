# Deep Reinforcement Learning 
**Documentation: https://spinningup.openai.com/en/latest/**

## Action Spaces
- Discrete Action Spaces (i.e. Atari/Go)
    - This is effectively where only a finite number of moves are available to the agent
- Continuous Action Spaces
    - This is where actions are real-valued vectors

## Policies
- A deterministic policy is a function in the form `𝜋𝕕:𝑆→𝐴`. This is a function from the set of states in the environment `S`, to the set of actions `A`. 
- A stochastic policy can be represented as a family of conditional probability distribution `𝜋𝕤(𝐴∣𝑆)`, from the set of states, `𝑆` to the set of actions, `𝐴` 
- In some particular case of games of change (e.g. poker), there are sources of randomness, a deterministic policy might not always be appropriate. For example, in poker, not all information (e.g. the cards of the other players) is available. In those circumstances, the agent might decide to play differently depending on the round (time step). More concretely, the agent could decide to go "all-in" 2/3 of the times whenever it has a hand with two aces and there are two uncovered aces on the table and decide to just "raise" 1/3 of the other times.
- **A deterministic policy can be interpreted as a stochastic policy that gives the probability of 1 to one of the available actions (and 0 to the remaining actions), for each state.**

## Rewards and Returns
- The reward function outputs a reward giving a state action pair as it's input.
- The goal of the agent/policy is to maximise some notion of cumulative reward over a trajectory (a sequence of states and actions in the world) 
- Some kind of returns are:
    - **Finite-Horizon Undiscounted Return** - The sum of rewards obtained in a fixed window of steps.
    - **Infinite-Horizon Discounted Return** - The sum of all rewards ever obtained by the agent, but discounted by how far off in the future they're obtained. We add a discount factor and under reasonable conditions, all infinite sums with a discount factor will converge. 

## PPO
- Actor critic methods are sensitive to pertubations
- PPO addresses this by limiting the updates to the policy network
    - Base the update on the ratio of new policy to old
- Have to account for goodness of state
- Keeps track of a fixed length trajectory of memories
- Uses multiple network updates per data sample
    - Minibatch stochastic gradient ascent
- Can also use multiple parallel actors (CPU)

We will be using two distinct networks instead of shared inputs
The critic network evaluates the states and loss function
Actor decides what to do based on the current state
- Network outputs probabilities (softmasx) for a distribution
- Exploration due to nature of distribution

### PPO Memory
Memory is fixed to length T (20 steps for cartpole) 

We will be tracking:
- States
- Actions
- Rewards
- Dones
- Values
- Log Probs

