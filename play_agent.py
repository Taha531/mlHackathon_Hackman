import time
import os
from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv
from train_agent import RLAgent

if __name__ == "__main__":
    print("🤖 Testing trained RL Agent...")

    test_file = "C:/Users/taham/Documents/College/ML_Hackathon/Hackman/Data/test.txt"
    qtable_path = "C:/Users/taham/Documents/College/ML_Hackathon/Hackman/q_table.pkl"

    if not os.path.exists(test_file):
        print(f"[!] Test file not found at: {test_file}")
        exit()
    if not os.path.exists(qtable_path):
        print(f"[!] Q-table not found: {qtable_path}")
        exit()

    oracle = HMMOracle()
    env = HangmanEnv(test_file)
    agent = RLAgent(oracle)
    agent.load(qtable_path)

    print("[+] Q-table loaded successfully.")
    total_games = 2000
    wins = 0

    for game in range(1, total_games + 1):
        masked = env.reset()
        guessed = set()
        print(f"\n🎮 Game {game} — Word length: {len(env.word)}")
        env.display()

        while not env.done:
            time.sleep(0.2)
            prev_state = masked
            action = agent.choose_action(masked, guessed)
            masked, reward, done = env.step(action)
            guessed.add(action)
            env.display()

            # Optional online learning (comment out if not needed)
            agent.update_q(prev_state, action, reward, masked)

        if '_' not in masked:
            print(f"✅ Agent won! Word: {env.word.upper()}")
            wins += 1
        else:
            print(f"❌ Agent lost! Word: {env.word.upper()}")

        # Save after each game (optional)
        agent.save(qtable_path)

    print(f"\n🏁 Final Score: {wins}/{total_games} wins.")
