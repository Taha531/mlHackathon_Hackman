import time
import os
from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv
from train_agent import RLAgent

if __name__ == "__main__":
    print("🤖 Starting Hangman Agent test...")

    test_file = "C:/Users/taham/Documents/College/ML_Hackathon/Hackman/Data/test.txt"
    qtable_path = "C:/Users/taham/Documents/College/ML_Hackathon/Hackman/q_table.pkl"

    if not os.path.exists(test_file):
        print(f"[!] Test file not found at: {test_file}")
        exit()

    oracle = HMMOracle()
    env = HangmanEnv(test_file)
    agent = RLAgent(oracle)

    if not os.path.exists(qtable_path):
        print(f"[!] Q-table not found at: {qtable_path}")
        print("Make sure you've trained the agent first using train_agent.py.")
        exit()
    else:
        agent.load(qtable_path)
        print("[+] Q-table loaded successfully.")

    total_games = 5
    wins = 0

    print(f"\n🎯 Playing {total_games} games using the trained agent...\n")

    for game in range(1, total_games + 1):
        masked = env.reset()
        guessed = set()

        print(f"\n🎮 Game {game} — Word length: {len(env.word)}")
        print("-" * 40)
        env.display()

        # safety to avoid infinite loops
        for step in range(20):
            if env.done:
                break
            action = agent.choose_action(masked, guessed)
            masked, reward, done = env.step(action)
            guessed.add(action)
            env.display()
            time.sleep(0.3)

        if '_' not in masked:
            print(f"✅ Agent won! Word: {env.word.upper()}")
            wins += 1
        else:
            print(f"❌ Agent lost! Word: {env.word.upper()}")

    print(f"\n🏁 Final Score: {wins}/{total_games} wins.")
