'''
 Simple HMM-like Oracle for Hangman
 Uses your cleaned and length-wise dissected corpus
 to estimate letter probabilities for each hidden word.
'''

import os
import collections
import matplotlib.pyplot as plt   # <--- new import for plotting

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

class HMMOracle:
    def __init__(self, grouped_folder='C:/Users/taham/Documents/College/ML_Hackathon/Hackman/length_wise_disection'):
        self.grouped_folder = grouped_folder
        self.words_by_len = {}
        self.pos_counts = {}
        self.letter_counts = collections.Counter()
        self.bigram_counts = {}
        self._load_data()
        self._build_stats()

    def _load_data(self):
        if not os.path.isdir(self.grouped_folder):
            raise FileNotFoundError(f"Folder '{self.grouped_folder}' not found.")
        for fname in os.listdir(self.grouped_folder):
            if fname.startswith("len") and fname.endswith(".txt"):
                try:
                    L = int(fname[3:-4])
                except:
                    continue
                path = os.path.join(self.grouped_folder, fname)
                with open(path, 'r') as f:
                    words = [w.strip().lower() for w in f if w.strip().isalpha()]
                if words:
                    self.words_by_len[L] = words

    def _build_stats(self):
        for L, words in self.words_by_len.items():
            pos_counters = [collections.Counter() for _ in range(L)]
            bigram_counter = collections.Counter()
            for w in words:
                for i, ch in enumerate(w):
                    pos_counters[i][ch] += 1
                    self.letter_counts[ch] += 1
                for i in range(len(w) - 1):
                    bigram_counter[(w[i], w[i + 1])] += 1
            self.pos_counts[L] = pos_counters
            self.bigram_counts[L] = bigram_counter
        self.total_letters = sum(self.letter_counts.values())

    def _match_candidates(self, masked_word, guessed):
        L = len(masked_word)
        if L not in self.words_by_len:
            return []
        candidates = []
        words = self.words_by_len[L]
        for w in words:
            ok = True
            for i, ch in enumerate(masked_word):
                if ch != '_' and w[i] != ch:
                    ok = False
                    break
            if not ok:
                continue
            for g in guessed:
                if g in w and g not in masked_word:
                    ok = False
                    break
            if ok:
                candidates.append(w)
        return candidates

    def get_probs(self, masked_word, guessed_set):
        masked_word = masked_word.lower()
        guessed_set = set(guessed_set)
        L = len(masked_word)
        if L not in self.words_by_len:
            return self._fallback(guessed_set)

        candidates = self._match_candidates(masked_word, guessed_set)
        if not candidates:
            candidates = self.words_by_len[L]

        letter_counts = collections.Counter()
        blanks = [i for i, c in enumerate(masked_word) if c == '_']
        for w in candidates:
            for i in blanks:
                letter_counts[w[i]] += 1

        # smoothing and fallback
        for ch in ALPHABET:
            letter_counts[ch] += 1
        for g in guessed_set:
            letter_counts[g] = 0

        total = sum(letter_counts.values())
        return {ch: letter_counts[ch] / total for ch in ALPHABET}

    def _fallback(self, guessed):
        total = sum(self.letter_counts.values()) + 26
        probs = {}
        for ch in ALPHABET:
            if ch in guessed:
                probs[ch] = 0
            else:
                probs[ch] = (self.letter_counts.get(ch, 0) + 1) / total
        s = sum(probs.values())
        for ch in probs:
            probs[ch] /= s
        return probs
    

oracle = HMMOracle()
masked = '__pp_e'
guessed = {'p', 'e'}

probs = oracle.get_probs(masked, guessed)
top = sorted(probs.items(), key=lambda x: -x[1])[:8]
print("\nTop letter predictions for", masked, ":", top)

# Plot the full probability distribution
letters = list(probs.keys())
values = [probs[ch] for ch in letters]

plt.figure(figsize=(12, 5))
plt.bar(letters, values, color='skyblue')
plt.title(f"Letter Probability Distribution for '{masked}' (guessed: {', '.join(guessed)})")
plt.xlabel("Letters")
plt.ylabel("Probability")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
