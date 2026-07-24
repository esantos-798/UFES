from experiments.experiment_grid import generate_grid
# Importa o runner que gerencia a lista inteira
from experiments.experiment_runner import ExperimentRunnerGrid

def main():
    # 1. Gera a lista com os 7 experimentos originais
    experiments = generate_grid()

    print()
    print("=" * 80)
    print(f"Experimentos encontrados: {len(experiments)}")
    print("=" * 80)

    # 2. CORREÇÃO: Passa a lista 'experiments' DIRETO, sem o laço 'for'
    grid = ExperimentRunnerGrid(experiments)

    # 3. Executa a lista toda internamente
    grid.run()

if __name__ == "__main__":
    main()