# 🧠 Hackman — Reinforcement Learning + HMM Hangman Agent

## 📘 Overview
Hackman is an intelligent Hangman-playing agent built using **Hidden Markov Models (HMM)** and **Reinforcement Learning (RL)**.  
The system learns word structures from a 50,000-word corpus and trains an RL agent to guess letters efficiently, balancing exploration and exploitation.

---

## 🧩 Project Structure
```
Hackman/
│
├── Data/
│   ├── corpus.txt               # Original dataset (50,000 words)
│   ├── test.txt                 # Hidden test dataset
│
├── length_wise_disection/       # Cleaned, grouped words by length
│
├── data_clean.py                # Cleans and groups dataset
├── hmm_oracle.py                # Builds the HMM oracle (letter probability model)
├── hangman_env.py               # Game environment for training/testing
├── train_agent.py               # Q-learning or DQN RL agent training
├── play_agent.py                # Evaluation script (2000 games, scoring)
├── q_table.pkl                  # Saved RL model (Q-table)
│
├── Analysis_Report.pdf          # Final analysis and insights (to be generated)
├── README.md                    # This file
```

---

## ⚙️ Step-by-Step Workflow

### 1️⃣ Data Cleaning
Run:
```bash
python data_clean.py
```
Creates:
- `cleaned_corpus.txt`
- `length_wise_disection/` folder grouping words by length

This ensures consistent casing, removes duplicates, and organizes words logically for the HMM.

---

### 2️⃣ Hidden Markov Model (HMM)
Run:
```bash
python hmm_oracle.py
```
Builds a probability model that predicts letter likelihoods for partially known words.  
- Hidden states ≈ letter positions  
- Emissions ≈ actual letters  
- Output: `get_probs(masked_word, guessed)` returns a probability distribution over the alphabet.

---

### 3️⃣ Reinforcement Learning Agent
Run:
```bash
python train_agent.py
```
Trains a **Q-learning** agent to play Hangman:
- **State** = (masked_word, guessed_letters)
- **Actions** = available letters (a–z)
- **Rewards:**
  - +10 for correct guess
  - -1 for wrong guess
  - -5 for repeated guesses
  - +100 for win, -50 for loss
- **Exploration vs. Exploitation:** ε-greedy with decaying ε.

---

### 4️⃣ Evaluation (Test Phase)
Run:
```bash
python play_agent.py
```
- Plays 2000 test games.
- Does **not modify** the Q-table (read-only).
- Displays per-game and final stats:
  - Success rate
  - Wrong guesses
  - Repeated guesses
  - Final Score:
    ```
    (SuccessRate * 2000) - (WrongGuesses * 5) - (RepeatedGuesses * 2)
    ```

---

## 📊 Outputs Collected for Report
You will later provide:
- **Training logs** (`reward per episode`, `ε decay`)
- **Evaluation summary** (wins, score)
- **Plots** of learning curve and performance  
These will go into `Analysis_Report.pdf`.

---

## 🧪 Requirements
```
Python ≥ 3.10
numpy
pandas
matplotlib
torch (optional if using GPU/DQN)
```

Install:
```bash
pip install numpy pandas matplotlib torch
```

---

## 🧠 Model Summary
| Component | Type | Description |
|------------|------|-------------|
| HMM Oracle | Statistical Model | Learns letter emission probabilities from corpus |
| RL Agent | Q-learning / DQN | Learns optimal guessing strategy |
| Environment | Hangman Simulation | Provides state, reward, and word feedback |

---

## 🎯 Scoring Formula
```
Final Score = (SuccessRate * 2000) - (WrongGuesses * 5) - (RepeatedGuesses * 2)
```

---

## 🧩 Future Work
- Move from Q-table → Deep Q-Network (DQN)
- Use multi-length adaptive HMMs
- Integrate GPU-accelerated neural training
- Add reward shaping for partial progress
