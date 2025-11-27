import sys
import argparse
import csv
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Ensure Planet Wars Python bindings are on the path
PW_PYTHON_PATH = "planet-wars-rts/app/src/main/python"
if PW_PYTHON_PATH not in sys.path:
    sys.path.insert(0, PW_PYTHON_PATH)

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.game_runner import GameRunner  # type: ignore
from core.game_state import GameParams, Player  # type: ignore
from agents.random_agents import PureRandomAgent, CarefulRandomAgent  # type: ignore
from agents.greedy_heuristic_agent import GreedyHeuristicAgent  # type: ignore
from sharp_agent import SharpAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark games between agents and save results to CSV.")
    parser.add_argument("--agent1", type=str, choices=["pure", "careful", "greedy", "sharp"], default="pure", help="Type of agent 1: 'pure', 'careful', 'greedy', or 'sharp' (default: pure).")
    parser.add_argument("--agent2", type=str, choices=["pure", "careful", "greedy", "sharp"], default="greedy", help="Type of agent 2: 'pure', 'careful', 'greedy', or 'sharp' (default: greedy).")
    parser.add_argument("--n-games", type=int, default=100000, help="Number of games to run (default: 100000)")
    parser.add_argument("--num-planets", type=int, default=12, help="Number of planets in the map (default: 12)")
    return parser.parse_args()

def make_agent(kind: str):
    if kind == "pure":
        return PureRandomAgent()
    if kind == "careful":
        return CarefulRandomAgent()
    if kind == "greedy":
        return GreedyHeuristicAgent()
    if kind == "sharp":
        return SharpAgent()
    raise ValueError(f"Unknown agent type: {kind}")


def run_single_game(job):
    """Run a single game and return the CSV row data."""
    game_index, agent1_kind, agent2_kind, num_planets = job

    agent1 = make_agent(agent1_kind)
    agent2 = make_agent(agent2_kind)
    game_params = GameParams(num_planets=num_planets)
    runner = GameRunner(agent1, agent2, game_params)

    final_model = runner.run_game()
    winner = final_model.get_leader()

    planets = final_model.state.planets
    p1_planets = sum(1 for p in planets if p.owner == Player.Player1)
    p2_planets = sum(1 for p in planets if p.owner == Player.Player2)
    neutral_planets = sum(1 for p in planets if p.owner == Player.Neutral)

    p1_ships = final_model.get_ships(Player.Player1)
    p2_ships = final_model.get_ships(Player.Player2)

    return [game_index, str(winner), p1_planets, p2_planets, neutral_planets, p1_ships, p2_ships]

def main():
    args = parse_args()

    # Fixed number of worker processes
    n_workers = max(1, min(12, args.n_games))

    print(f"Benchmark: {args.agent1} vs {args.agent2}")
    print(f"Games: {args.n_games}, Num planets: {args.num_planets}")
    print(f"Using {n_workers} worker processes")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"{timestamp}_{args.agent1}_v_{args.agent2}_benchmark.csv"
    if not os.path.isabs(outfile):
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), outfile)

    start_time = time.time()

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "winner", "p1_planets", "p2_planets", "neutral_planets", "p1_ships", "p2_ships"])

        jobs = ((i, args.agent1, args.agent2, args.num_planets) for i in range(1, args.n_games + 1))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for completed, row in enumerate(executor.map(run_single_game, jobs), start=1):
                writer.writerow(row)
                print(f"Completed {completed}/{args.n_games} games")

    time_diff = time.time() - start_time

    print("=" * 50)
    print(f"Results saved to {outfile}")
    print(f"Total time: {time_diff:.2f} seconds "
          f"({time_diff/args.n_games:.4f} s/game)")

if __name__ == "__main__":
    main()
