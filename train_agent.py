import random
import pickle
from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

class QLearningAgent:
    def __init__(self, oracle, alpha=0.1, gamma=0.9, epsilon=1.0, eps_decay=0.995, eps_min=0.05):
        self.oracle = oracle
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.q_table = {}

    def get_state(self, masked, guessed):
        return masked + "|" + "".join(sorted(guessed))

    def choose_action(self, masked, guessed):
        state = self.get_state(masked, guessed)
        if random.random() < self.epsilon:
            # bias exploration with HMM probabilities
            probs = self.oracle.get_probs(masked, guessed)
            return max(probs, key=probs.get)
        q_vals = self.q_table.get(state, {})
        if not q_vals:
            return random.choice([ch for ch in ALPHABET if ch not in guessed])
        return max(q_vals, key=q_vals.get)

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        old_value = self.q_table[state][action]
        next_max = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            next_max = max(self.q_table[next_state].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"✅ Q-table saved to {path}")

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"✅ Q-table loaded from {path}")


if __name__ == "__main__":
    corpus_file = "cleaned_corpus.txt"
    env = HangmanEnv(corpus_file)
    oracle = HMMOracle()
    agent = QLearningAgent(oracle)

    EPISODES = 5000
    for ep in range(1, EPISODES + 1):
        masked = env.reset()
        guessed = set()
        total_reward = 0

        while not env.done:
            action = agent.choose_action(masked, guessed)
            next_masked, reward, done = env.step(action)
            guessed.add(action)
            next_state = agent.get_state(next_masked, guessed)
            state = agent.get_state(masked, guessed)
            agent.learn(state, action, reward, next_state)
            masked = next_masked
            total_reward += reward

        agent.decay_epsilon()
        if ep % 100 == 0:
            print(f"Ep {ep} | eps={agent.epsilon:.3f} | reward={total_reward}")

    agent.save()
