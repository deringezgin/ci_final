import glob
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

MAX_GENERATIONS = 500

plt.style.use(['science', 'no-latex'])

def read_generation_stats(db_path):
    connection = sqlite3.connect(db_path)  # Connect to the db
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT generation,
                   AVG(fitness) AS avg_fitness,
                   MAX(fitness) AS best_fitness
            FROM results
            GROUP BY generation
            ORDER BY generation
            """
        )
        # Get the generational data for each generation in the run
        rows = cursor.fetchall()[:MAX_GENERATIONS]
        generations = [int(r[0]) for r in rows]
        avg_fitness = [float(r[1]) for r in rows]
        best_fitness = [float(r[2]) for r in rows]
        return generations, avg_fitness, best_fitness
    finally:
        connection.close()  # At the end close the database connection

def read_run_config(db_path, key, default=""):
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT v FROM config WHERE k = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else default
    finally:
        connection.close()

def plot_run(db_path, output_dir):
    run_id = Path(db_path).stem
    layers = read_run_config(db_path, "hidden_sizes", "[]")
    games_per_individual = read_run_config(db_path, "games_per_eval", "")

    generations, avg_fitness, best_fitness = read_generation_stats(db_path)
    if not generations:
        return [], []

    # Convert to percentages
    avg_pct = [x * 100.0 for x in avg_fitness]
    best_pct = [x * 100.0 for x in best_fitness]

    # Plot the run
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(generations, avg_pct, label="Average fitness", linewidth=2)
    ax.plot(generations, best_pct, label="Best fitness", linewidth=2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Win Percentage")
    ax.set_ylim(0, 100)
    ax.set_title(f"Run: {run_id}, {layers} - {games_per_individual}")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"run_{run_id}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return generations, avg_pct

def main():
    project_root_folder = Path(__file__).resolve().parent
    data_dir = project_root_folder / "data"
    plots_dir = project_root_folder / "plots"

    db_files = sorted(glob.glob(str(data_dir / "*.sqlite3")))

    all_generations = []
    all_avg_series = []

    for db_path in db_files:
        print("Processing:", db_path)
        generations, avg_pct = plot_run(db_path, plots_dir)
        if generations:
            all_generations.append(generations)
            all_avg_series.append(avg_pct)

    # Plot mean of the average fitness across all runs, with 25–75 percentile areas
    if all_avg_series:
        # Crop all the series to the shortest length
        min_len = min(len(g) for g in all_generations)
        common_generations = all_generations[0][:min_len]
        trimmed_series = [series[:min_len] for series in all_avg_series]

        arr = np.array(trimmed_series, dtype=float)
        mean_avg_pct = arr.mean(axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(common_generations, mean_avg_pct, label="Mean of Average Fitness Across Runs", linewidth=1.5)
        ax.fill_between(common_generations, p25, p75, color="blue", alpha=0.2, label="25-75 percentile of average fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Win Percentage")
        ax.set_ylim(0, 100)
        ax.set_title("Mean Average Fitness Across 10 Independent Runs")
        ax.legend()

        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / "mean_avg_fitness.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
