import os
import sys
import traceback
from hmm_oracle import HMMOracle
from hangman_env import HangmanEnv
from train_agent import QLearningAgent

def safe_print(*args, **kwargs):
    """Ensure immediate flush so output appears in realtime."""
    print(*args, **kwargs)
    sys.stdout.flush()

def check_paths(test_file, q_path, hmm_folder_check=None):
    ok = True
    if not os.path.exists(test_file):
        safe_print(f"[ERROR] Test file not found: {test_file}")
        ok = False
    if not os.path.exists(q_path):
        safe_print(f"[ERROR] Q-table file not found: {q_path}")
        ok = False
    if hmm_folder_check and not os.path.isdir(hmm_folder_check):
        safe_print(f"[WARNING] HMM folder not found (may still work): {hmm_folder_check}")
    return ok

if __name__ == "__main__":
    try:
        safe_print("🤖 Starting evaluation (read-only mode)...\n")

        base_dir = os.path.abspath(os.getcwd())
        test_file = os.path.join(base_dir, "Data", "test.txt")
        q_path = os.path.join(base_dir, "q_table.pkl")
        hmm_folder_default = os.path.join(base_dir, "length_wise_disection")

        if not check_paths(test_file, q_path, hmm_folder_default):
            safe_print("Fix the missing files and rerun. Exiting.")
            sys.exit(1)

        oracle = HMMOracle()
        safe_print("[OK] HMMOracle loaded.")

        env = HangmanEnv(test_file)
        safe_print(f"[OK] HangmanEnv loaded with {len(env.words)} words.")

        agent = QLearningAgent(oracle)
        agent.load(q_path)
        safe_print(f"[OK] Q-table loaded with {len(agent.q_table)} states.\n")

        agent.epsilon = 0.0  # evaluation only

        total_games = 2000
        wins = 0
        total_wrong = 0
        total_repeated = 0
        total_score = 0

        safe_print(f"{'Game':<6} {'Result':<6} {'Len':<5} {'Wrong':<6} {'Repeat':<8} {'Score':<6}")

        def choose_action_fallback(agent, masked, guessed):
            """Use Q-table if available, else fallback to HMM probabilities."""
            state = agent.get_state(masked, guessed)
            qvals = agent.q_table.get(state, {})
            if qvals:
                choices = [(a, v) for a, v in qvals.items() if a not in guessed]
                if choices:
                    return max(choices, key=lambda x: x[1])[0]
            # fallback to HMM oracle
            probs = agent.oracle.get_probs(masked, guessed)
            letters = [c for c in probs if c not in guessed]
            letters.sort(key=lambda c: -probs[c])
            return letters[0] if letters else 'a'

        for game_id in range(1, total_games + 1):
            masked = env.reset()
            guessed = set()
            wrong_guesses = 0
            repeated_guesses = 0
            game_score = 0

            while not env.done:
                action = choose_action_fallback(agent, masked, guessed)
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
            result = "WIN" if "_" not in masked else "LOSS"
            if result == "WIN":
                wins += 1

            safe_print(f"{game_id:<6} {result:<6} {len(env.word):<5} {wrong_guesses:<6} {repeated_guesses:<8} {int(game_score):<6}")

        success_rate = wins / total_games
        final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)

        safe_print("\n🏁 FINAL EVALUATION SUMMARY")
        safe_print(f"✅ Wins: {wins}/{total_games} ({success_rate * 100:.2f}%)")
        safe_print(f"❌ Total Wrong Guesses: {total_wrong}")
        safe_print(f"🔁 Total Repeated Guesses: {total_repeated}")
        safe_print(f"💰 Average Reward/Game: {total_score / total_games:.2f}")
        safe_print(f"🏆 Final Score: {final_score:.2f}")

    except Exception:
        safe_print("[FATAL] Unexpected error during evaluation:")
        traceback.print_exc()
        sys.exit(1)
