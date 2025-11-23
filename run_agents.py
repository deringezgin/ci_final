import sys
import argparse
import importlib

PW_PYTHON_PATH = "planet-wars-rts/app/src/main/python"
if PW_PYTHON_PATH not in sys.path:
	sys.path.insert(0, PW_PYTHON_PATH)

from core.game_runner import GameRunner  # type: ignore
from core.game_state import GameParams, Player  # type: ignore
from core.forward_model import ForwardModel  # type: ignore

def load_agent(class_path: str):
	"""
	Load an agent class using the following format: module.ClassName
	This would be used as "from module import ClassName"
	"""
	if "." not in class_path:
		raise ValueError(f"Invalid class path '{class_path}'. Use the module.ClassName format (for instance agents.random_agents.CarefulRandomAgent).")
	mod_name, cls_name = class_path.rsplit(".", 1)
	mod = importlib.import_module(mod_name)
	cls = getattr(mod, cls_name)
	return cls()

def parse_args():
    """Argument parser for the program"""
    parser = argparse.ArgumentParser(description="Run Planet Wars matches between two agents.")
    parser.add_argument("--agent1", type=str, default="sharp_agent.SharpAgent", help="Agent 1 class (module.ClassName). Default: sharp_agent.SharpAgent")
    parser.add_argument("--agent2", type=str, default="agents.greedy_heuristic_agent.GreedyHeuristicAgent", help="Agent 2 class (module.ClassName). Default: agents.greedy_heuristic_agent.GreedyHeuristicAgent")
    parser.add_argument("--n-games", type=int, default=200, help="Number of games to run. Default: 200")
    parser.add_argument("--num-planets", type=int, default=12, help="Number of planets in the map. Default: 12")
    return parser.parse_args()

def main():
	args = parse_args()   # Parse the commnad line arguments
 
	# Load the agents from the given class paths
	agent1 = load_agent(args.agent1)
	agent2 = load_agent(args.agent2)
 
	print("AGENT 1:", agent1.get_agent_type())
	print("AGENT 2:", agent2.get_agent_type())
	print("=" * 50)
	# Configure the game parameters with the given number of planets
	game_params = GameParams(num_planets=args.num_planets)
	runner = GameRunner(agent1, agent2, game_params)

	# Run the games and count the wins for each player
	scores = {Player.Player1: 0, Player.Player2: 0, Player.Neutral: 0}
	for i in range(args.n_games):
		final_model = runner.run_game()
		winner = final_model.get_leader()
		scores[winner] += 1

		planets = final_model.state.planets
		p1_planets = sum(1 for p in planets if p.owner == Player.Player1)
		p2_planets = sum(1 for p in planets if p.owner == Player.Player2)
		neutral_planets = sum(1 for p in planets if p.owner == Player.Neutral)

		p1_ships = final_model.get_ships(Player.Player1)
		p2_ships = final_model.get_ships(Player.Player2)

		print(f"Game {i+1}/{args.n_games} winner: {winner} (P1 planets: {p1_planets}, P2 planets: {p2_planets}, Neutral: {neutral_planets}, P1 ships: {p1_ships:.1f}, P2 ships: {p2_ships:.1f})")

	# Print the results of the games
	print("=" * 50)
	print(f"Agent1: {agent1.get_agent_type()}  vs  Agent2: {agent2.get_agent_type()}")
	print("Results (wins):", scores)
	if ForwardModel.n_updates > 0:
		print(f"Successful actions: {ForwardModel.n_actions}")
		print(f"Failed actions: {ForwardModel.n_failed_actions}")

if __name__ == "__main__":
	main()
