from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv
from train_agent import QLearningAgent

if __name__ == "__main__":
    print("🤖 Evaluating Q-Learning Agent...")

    test_file = "Data/test.txt"  
    oracle = HMMOracle()
    env = HangmanEnv(test_file)
    agent = QLearningAgent(oracle)
    agent.load("q_table.pkl")

    total_games = 2000  
    wins = 0
    total_wrong = 0
    total_repeated = 0
    total_score = 0

    for _ in range(total_games):
        masked = env.reset()
        guessed = set()
        wrong_guesses = 0
        repeated_guesses = 0
        game_score = 0

        while not env.done:
            action = agent.choose_action(masked, guessed)
            if action in guessed:
                repeated_guesses += 1
            masked, reward, done = env.step(action)
            guessed.add(action)
            if reward == -1:
                wrong_guesses += 1
            game_score += reward

        total_score += game_score
        total_wrong += wrong_guesses
        total_repeated += repeated_guesses
        if "_" not in masked:
            wins += 1

    success_rate = wins / total_games
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)

    print("\n🏁 EVALUATION SUMMARY")
    print(f"✅ Wins: {wins}/{total_games} ({success_rate * 100:.2f}%)")
    print(f"❌ Wrong Guesses: {total_wrong}")
    print(f"🔁 Repeated Guesses: {total_repeated}")
    print(f"💰 Average Reward per Game: {total_score / total_games:.2f}")
    print(f"🏆 Final Score: {final_score:.2f}")
