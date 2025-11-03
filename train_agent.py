import random
import pickle
import os
from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv

class RLAgent:
    def __init__(self, oracle, alpha=0.3, gamma=0.8, epsilon=0.1):
        self.oracle = oracle
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}

    def get_state_key(self, masked, guessed):
        # simplified representation
        return masked

    def choose_action(self, masked, guessed):
        """Epsilon-greedy action selection guided by HMM probabilities."""
        state_key = self.get_state_key(masked, guessed)

        available = [ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch not in guessed]

        # Exploration vs. exploitation
        if random.random() < self.epsilon:
            return random.choice(available)

        # Get letter probabilities from HMM
        probs = self.oracle.get_probs(masked, guessed)
        sorted_letters = sorted(probs.items(), key=lambda x: -x[1])

        # Prefer high-probability letters not yet guessed
        for ch, _ in sorted_letters:
            if ch not in guessed:
                return ch
        return random.choice(available)

    def update_q(self, state, action, reward, next_state):
        s_key = state
        ns_key = next_state

        # Initialize if not present
        self.q_table.setdefault(s_key, {a: 0 for a in "abcdefghijklmnopqrstuvwxyz"})
        self.q_table.setdefault(ns_key, {a: 0 for a in "abcdefghijklmnopqrstuvwxyz"})

        old_value = self.q_table[s_key][action]
        next_max = max(self.q_table[ns_key].values())

        # Q-learning update rule
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[s_key][action] = new_value

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="q_table.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)


if __name__ == "__main__":
    print("🔁 Training RL Agent using HMM Oracle...")

    corpus_file = "C:/Users/taham/Documents/College/ML_Hackathon/Hackman/cleaned_corpus.txt"
    oracle = HMMOracle()
    env = HangmanEnv(corpus_file)
    agent = RLAgent(oracle)

    EPISODES = 3000
    stats = {"wins": 0, "losses": 0}
    performance = []

    for ep in range(1, EPISODES + 1):
        masked = env.reset()
        guessed = set()
        start_state = masked

        while not env.done:
            action = agent.choose_action(masked, guessed)
            prev_state = masked

            masked, reward, done = env.step(action)
            guessed.add(action)

            agent.update_q(prev_state, action, reward, masked)

        # Track stats
        if '_' not in masked:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        # Print progress every 100 episodes
        if ep % 100 == 0:
            total = stats["wins"] + stats["losses"]
            win_rate = stats["wins"] / total if total else 0
            performance.append(win_rate)
            print(f"Episode {ep}: Win rate = {win_rate:.2f}, Epsilon = {agent.epsilon:.3f}")
            agent.save()

    print("\n✅ Training Complete!")
    print(f"Total Wins: {stats['wins']} | Total Losses: {stats['losses']}")
    print("Q-table saved as q_table.pkl")
