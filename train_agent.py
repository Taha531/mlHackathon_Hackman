import random
import pickle
import os
from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv

class RLAgent:
    def __init__(self, oracle, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.oracle = oracle
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_state_key(self, masked, guessed):
        return (masked, ''.join(sorted(guessed)))

    def choose_action(self, masked, guessed):
        """Choose letter based on epsilon-greedy and HMM probabilities."""
        state_key = self.get_state_key(masked, guessed)

        if random.random() < self.epsilon:
            # Explore: random unused letter
            available = [ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch not in guessed]
            return random.choice(available)

        # Exploit: use oracle probabilities
        probs = self.oracle.get_probs(masked, guessed)
        sorted_letters = sorted(probs.items(), key=lambda x: -x[1])
        for ch, _ in sorted_letters:
            if ch not in guessed:
                return ch
        return random.choice("abcdefghijklmnopqrstuvwxyz")

    def update_q(self, state, action, reward, next_state):
        """Standard Q-learning update."""
        s_key = self.get_state_key(*state)
        ns_key = self.get_state_key(*next_state)
        self.q_table.setdefault(s_key, {a: 0 for a in "abcdefghijklmnopqrstuvwxyz"})
        self.q_table.setdefault(ns_key, {a: 0 for a in "abcdefghijklmnopqrstuvwxyz"})

        old_value = self.q_table[s_key][action]
        next_max = max(self.q_table[ns_key].values())

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

    EPISODES = 300
    stats = {"wins": 0, "losses": 0}

    for ep in range(1, EPISODES + 1):
        masked = env.reset()
        guessed = set()

        while not env.done:
            action = agent.choose_action(masked, guessed)
            prev_state = (masked, guessed.copy())

            masked, reward, done = env.step(action)
            guessed.add(action)

            next_state = (masked, guessed.copy())
            agent.update_q(prev_state, action, reward, next_state)

        if '_' not in masked:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        # Progress update every 50 episodes
        if ep % 50 == 0:
            total = stats["wins"] + stats["losses"]
            win_rate = stats["wins"] / total if total > 0 else 0
            print(f"Episode {ep}: Win rate = {win_rate:.2f}")
            agent.save()

    print("\n✅ Training Complete!")
    print(f"Total Wins: {stats['wins']} | Total Losses: {stats['losses']}")
    print("Q-table saved as q_table.pkl")
