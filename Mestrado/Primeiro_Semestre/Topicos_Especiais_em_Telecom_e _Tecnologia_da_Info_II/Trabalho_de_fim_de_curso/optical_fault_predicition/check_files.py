from pathlib import Path

# Pega a primeira pasta de run que encontrar para inspecionar
runs_dir = Path("results/runs")
first_run = next(runs_dir.iterdir())

print(f"🔍 Inspecionando a estrutura da pasta: {first_run.name}\n")
print("📂 Arquivos encontrados lá dentro:")

for file in first_run.rglob("*"):
    if file.is_file():
        # Mostra o caminho relativo a partir de results/runs/nome_da_run
        relative_path = file.relative_to(first_run)
        # Exibe o tamanho do arquivo em MB para termos uma pista se é um array pesado
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {relative_path} ({size_mb:.2f} MB)")