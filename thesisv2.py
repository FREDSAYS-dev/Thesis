import numpy as np

class NPCPolicy:
    """
    NPC policy with a role (Ally, Competitive, Dialogue, etc.)
    and optional parameterized goal or policy function.
    """
    def __init__(self, role="Ally", policy_func=None, goal=None, name=None):
        self.role = role
        self.policy_func = policy_func or (lambda env, state: 0)  # default: always left
        self.goal = goal  # Can parameterize the agent's target if needed
        self.name = name or role

    def act(self, env, state):
        """
        Returns the action for this agent in the given environment and state.
        """
        return self.policy_func(env, state)

    def __repr__(self):
        return f"NPCPolicy(role={self.role}, name={self.name})"

class TwoAgentEnv:
    """
    Minimal 2-agent 1D world with agent roles.
    - positions: [pos_agent0, pos_agent1], start [0, 2]
    - actions per agent: 0=left, 1=right
    - goal at position 4
    - collision penalty if both land on same non-goal position
    """
    def __init__(self, agent_policies):
        assert len(agent_policies) == 2, "This environment expects exactly 2 agents."
        self.num_positions = 5
        self.goal = 4
        self.num_actions = 2
        self.agent_policies = agent_policies  # list of NPCPolicy or similar
        self.reset()

    def reset(self):
        self.positions = [0, 2]
        return self.positions.copy()

    def step(self):
        # Each agent chooses their action
        actions = [agent.act(self, pos) for agent, pos in zip(self.agent_policies, self.positions)]
        rewards = [0.0, 0.0]
        done = False

        # Update positions
        for i, action in enumerate(actions):
            if action == 0:
                self.positions[i] = max(0, self.positions[i] - 1)
            else:
                self.positions[i] = min(self.num_positions - 1, self.positions[i] + 1)

        # Collision penalty (if not at goal)
        if self.positions[0] == self.positions[1] and self.positions[0] != self.goal:
            rewards = [-1.0, -1.0]

        # Goal reward
        for i in range(2):
            if self.positions[i] == self.goal:
                rewards[i] += 1.0
                done = True

        # Add info: agent roles/tags and current positions
        info = [{"role": agent.role, "name": agent.name, "pos": pos} for agent, pos in zip(self.agent_policies, self.positions)]
        return self.positions.copy(), rewards, done, info

# --- Example policies for different roles ---

def ally_policy(env, state):
    """
    Ally: tries to move toward the goal (right).
    """
    return 1

def competitive_policy(env, state):
    """
    Competitive: tries to move away from goal (left).
    """
    return 0

def dialogue_policy(env, state):
    """
    Dialogue: random action, placeholder for more complex communication behavior.
    """
    return np.random.randint(2)

# --- Example usage ---

if __name__ == "__main__":
    # Define agents
    agent_main = NPCPolicy(role="Main", policy_func=ally_policy, name="main_agent")
    agent_ally = NPCPolicy(role="Ally", policy_func=ally_policy, name="ally_npc")
    agent_competitive = NPCPolicy(role="Competitive", policy_func=competitive_policy, name="competitor_npc")
    agent_dialogue = NPCPolicy(role="Dialogue", policy_func=dialogue_policy, name="dialogue_npc")

    # Example: main agent + competitor
    print("Main agent vs. Competitive NPC")
    env = TwoAgentEnv([agent_main, agent_competitive])
    state = env.reset()
    for step in range(10):
        state, rewards, done, info = env.step()
        print(f"Step {step+1}: State: {state}, Rewards: {rewards}, Info: {info}")
        if done:
            break

    print("\nMain agent + Ally NPC")
    env2 = TwoAgentEnv([agent_main, agent_ally])
    state = env2.reset()
    for step in range(10):
        state, rewards, done, info = env2.step()
        print(f"Step {step+1}: State: {state}, Rewards: {rewards}, Info: {info}")
        if done:
            break

    print("\nMain agent + Dialogue NPC")
    env3 = TwoAgentEnv([agent_main, agent_dialogue])
    state = env3.reset()
    for step in range(10):
        state, rewards, done, info = env3.step()
        print(f"Step {step+1}: State: {state}, Rewards: {rewards}, Info: {info}")
        if done:
            break
