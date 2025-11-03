import random
import os
import time

class HangmanEnv:
    def __init__(self, corpus_file, max_lives=6):
        self.corpus_file = corpus_file
        self.max_lives = max_lives
        self.word_list = self._load_words(corpus_file)
        self.reset()

    def _load_words(self, corpus_file):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"File '{corpus_file}' not found.")
        with open(corpus_file, 'r') as f:
            return [w.strip().lower() for w in f if w.strip().isalpha()]

    def reset(self):
        """Start a new game."""
        self.word = random.choice(self.word_list)
        self.masked = ['_'] * len(self.word)
        self.guessed = set()
        self.lives = self.max_lives
        self.done = False
        return ''.join(self.masked)

    def _reveal(self, letter):
        for i, ch in enumerate(self.word):
            if ch == letter:
                self.masked[i] = letter

    def step(self, letter):
        """Make a guess and return (new_state, reward, done)."""
        letter = letter.lower()
        if self.done:
            raise Exception("Game finished. Please reset().")

        if letter in self.guessed:
            return ''.join(self.masked), -0.5, self.done

        self.guessed.add(letter)

        if letter in self.word:
            self._reveal(letter)
            reward = 1
        else:
            self.lives -= 1
            reward = -1

        if '_' not in self.masked:
            reward += 10
            self.done = True
        elif self.lives <= 0:
            reward -= 5
            self.done = True

        return ''.join(self.masked), reward, self.done

    def get_state(self):
        return ''.join(self.masked), self.guessed

    def display(self):
        """Simple visual output for debugging or demo."""
        print(f"Word: {' '.join(self.masked)}")
        print(f"Lives: {self.lives} | Guessed: {', '.join(sorted(self.guessed))}")
        print("-" * 40)

    def __str__(self):
        return f"Word: {''.join(self.masked)} | Lives: {self.lives} | Guessed: {sorted(self.guessed)}"


# ---------------------------------------------------------------------
# 🧍 Manual Play Mode
# ---------------------------------------------------------------------
if __name__ == "__main__":
    corpus_file = "C:/Users/taham/Documents/College/ML_Hackathon/Hackman/cleaned_corpus.txt"
    env = HangmanEnv(corpus_file)
    state = env.reset()

    print("\n🎮 Welcome to Hangman!")
    print("Type one letter per turn. Type 'quit' to stop.\n")

    while not env.done:
        env.display()
        guess = input("Enter a letter: ").strip().lower()

        if guess == "quit":
            print("Exiting game...")
            break
        elif len(guess) != 1 or not guess.isalpha():
            print("Please enter a single alphabet letter.")
            continue

        state, reward, done = env.step(guess)
        print(f"Reward: {reward}")

        if done:
            env.display()
            if '_' not in env.masked:
                print(f"✅ You won! The word was: {env.word.upper()}")
            else:
                print(f"❌ You lost! The word was: {env.word.upper()}")
