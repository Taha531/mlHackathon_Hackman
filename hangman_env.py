import random

class HangmanEnv:
    def __init__(self, corpus_file):
        with open(corpus_file, 'r') as f:
            self.words = [w.strip().lower() for w in f if w.strip().isalpha()]
        self.reset()

    def reset(self):
        self.word = random.choice(self.words)
        self.masked = "_" * len(self.word)
        self.lives = 6
        self.guessed = set()
        self.done = False
        return self.masked

    def _reveal(self, letter):
        self.masked = ''.join([letter if self.word[i] == letter else self.masked[i] for i in range(len(self.word))])

    def step(self, letter):
        if self.done:
            return self.masked, 0, True

        if letter in self.guessed:
            return self.masked, -2, False

        self.guessed.add(letter)

        if letter in self.word:
            self._reveal(letter)
            reward = 2
        else:
            self.lives -= 1
            reward = -1

        if "_" not in self.masked:
            self.done = True
            reward += 10
        elif self.lives <= 0:
            self.done = True
            reward -= 5

        return self.masked, reward, self.done

    def display(self):
        print("Word:", " ".join(self.masked), "| Lives:", self.lives, "| Guessed:", sorted(list(self.guessed)))
