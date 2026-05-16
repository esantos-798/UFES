"""
=============================================================================
SIMULADOR DE ALARMES OTN — VERSÃO 2
=============================================================================

NOVIDADES:
    1. Resolução de papéis de nós baseada nos serviços e topologia
       - upstream_adjacente, downstream_adjacente: direto da posição
       - terminacao_odu_orig / terminacao_odu_dest: consultado no serviço
       - pass_through: não geram ODU-AIS/RDI, apenas OTS/OMS
    2. Suporte a dois modos: sem_correlacao e com_correlacao
    3. Label enriquecido com modo de correlação

Uso:
    python simulator.py --amostras 200 --output ../output/dataset.csv
=============================================================================
"""

import random
import uuid
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from alarm_rules_g798 import (TIPOS_ALARME, CENARIOS_FALHA,
                               ALARMES_ESPURIOS_GLOBAIS,
                               PROBABILIDADE_ESPURIO_GLOBAL)
from topology import NOS, ENLACES, SERVICOS, POSICOES_FALHA


# =============================================================================
# RESOLUÇÃO DE PAPÉIS DE NÓS
# =============================================================================

def resolver_papel(papel_no, posicao, servico):
    """
    Resolve o nome real do nó a partir do papel definido no cenário.

    Esta é a função central da versão 2. Ela consulta a posição de falha
    e o serviço afetado para determinar quem é cada ator na sequência
    de alarmes.

    Papéis possíveis:
        "upstream_adjacente"   → nó imediatamente antes da falha
        "downstream_adjacente" → nó imediatamente após da falha
        "terminacao_odu_orig"  → nó que termina o ODU no lado origem
        "terminacao_odu_dest"  → nó que termina o ODU no lado destino
        "local"                → o próprio nó em falha
        "remoto"               → par do nó em falha no serviço

    Args:
        papel_no: string com o papel (ex: "downstream_adjacente")
        posicao: dicionário da posição de falha
        servico: dicionário do serviço mais afetado

    Returns:
        Nome real do nó (ex: "NE3")
    """
    if papel_no == "upstream_adjacente":
        return posicao["upstream_adjacente"]

    elif papel_no == "downstream_adjacente":
        return posicao["downstream_adjacente"]

    elif papel_no == "terminacao_odu_orig":
        # Origem do serviço é sempre a primeira terminação ODU
        terminacoes = servico["terminacao_odu"]
        return terminacoes[0] if terminacoes else posicao["upstream_adjacente"]

    elif papel_no == "terminacao_odu_dest":
        # Destino do serviço é sempre a última terminação ODU
        terminacoes = servico["terminacao_odu"]
        return terminacoes[-1] if terminacoes else posicao["downstream_adjacente"]

    elif papel_no == "local":
        return posicao["nos_afetados"][0]

    elif papel_no == "remoto":
        nos = posicao["nos_afetados"]
        return nos[-1] if len(nos) > 1 else nos[0]

    else:
        # Já é um nome real de nó
        return papel_no


def selecionar_servico_principal(posicao):
    """
    Seleciona o serviço mais relevante para uma posição de falha.

    Quando múltiplos serviços são afetados, usamos o que tem mais
    nós no caminho (serviço mais longo = mais impactado).

    Args:
        posicao: dicionário da posição de falha

    Returns:
        Dicionário do serviço selecionado
    """
    servicos_afetados = posicao.get("servicos_afetados", [])
    if not servicos_afetados:
        # Fallback: usa o primeiro serviço disponível
        return list(SERVICOS.values())[0]

    # Seleciona o serviço com maior caminho (mais nós envolvidos)
    candidatos = [SERVICOS[s] for s in servicos_afetados if s in SERVICOS]
    return max(candidatos, key=lambda s: len(s["caminho"]))


# =============================================================================
# GERAÇÃO DE ALARMES
# =============================================================================

def gerar_alarme(tipo, timestamp_ms, no, direcao):
    """Cria um dicionário representando um alarme individual."""
    info = TIPOS_ALARME[tipo]
    return {
        "alarme_id":    str(uuid.uuid4())[:8],
        "tipo":         tipo,
        "nome":         info["nome"],
        "camada":       info["camada"],
        "severidade":   info["severidade"],
        "no":           no,
        "direcao":      direcao,
        "timestamp_ms": timestamp_ms,
    }


def inserir_espurios(sequencia, modo_config, tempo_max_ms):
    """
    Insere alarmes espúrios aleatórios na sequência.

    Args:
        sequencia: lista de alarmes já gerados
        modo_config: dicionário do modo (sem_correlacao ou com_correlacao)
        tempo_max_ms: tempo máximo para inserir espúrios

    Returns:
        Sequência com espúrios inseridos e reordenada
    """
    # Espúrios específicos do cenário/modo
    if random.random() < modo_config.get("probabilidade_espurio", 0):
        candidatos = modo_config.get("alarmes_possiveis_espurios", [])
        if candidatos:
            tipo_esp = random.choice(candidatos)
            ts_esp   = random.randint(0, max(tempo_max_ms, 1))
            no_esp   = random.choice(list(NOS.keys()))
            sequencia.append(gerar_alarme(tipo_esp, ts_esp, no_esp, "near-end"))

    # Espúrios globais
    if random.random() < PROBABILIDADE_ESPURIO_GLOBAL:
        tipo_esp = random.choice(ALARMES_ESPURIOS_GLOBAIS)
        ts_esp   = random.randint(0, max(tempo_max_ms, 1))
        no_esp   = random.choice(list(NOS.keys()))
        sequencia.append(gerar_alarme(tipo_esp, ts_esp, no_esp, "near-end"))

    sequencia.sort(key=lambda x: x["timestamp_ms"])
    return sequencia


# =============================================================================
# GERAÇÃO DE UMA AMOSTRA
# =============================================================================

def gerar_amostra(nome_cenario, nome_posicao, modo="sem_correlacao"):
    """
    Gera uma única amostra do dataset.

    Args:
        nome_cenario: chave em CENARIOS_FALHA (ex: 'corte_fibra')
        nome_posicao: chave em POSICOES_FALHA (ex: 'E2_NE2_NE3')
        modo: 'sem_correlacao' ou 'com_correlacao'

    Returns:
        Tupla (sequencia, features, label)
    """
    cenario    = CENARIOS_FALHA[nome_cenario]
    posicao    = POSICOES_FALHA[nome_posicao]
    modo_cfg   = cenario[modo]
    servico    = selecionar_servico_principal(posicao)
    sequencia  = []

    # -----------------------------------------------------------------
    # Gerar alarmes do cenário
    # -----------------------------------------------------------------
    for alarme_def in modo_cfg["alarmes"]:

        # Alarmes opcionais têm 60% de chance de aparecer
        if not alarme_def["obrigatorio"]:
            if random.random() > 0.6:
                continue

        # Aplica jitter
        jitter    = random.randint(-alarme_def["jitter_ms"],
                                    alarme_def["jitter_ms"])
        timestamp = max(0, alarme_def["delta_ms"] + jitter)

        # Resolve o nó real a partir do papel
        no_real = resolver_papel(alarme_def["papel_no"], posicao, servico)

        alarme = gerar_alarme(
            tipo        = alarme_def["tipo"],
            timestamp_ms = timestamp,
            no           = no_real,
            direcao      = alarme_def["direcao"],
        )
        sequencia.append(alarme)

    # -----------------------------------------------------------------
    # Inserir espúrios
    # -----------------------------------------------------------------
    tempo_max = max((a["timestamp_ms"] for a in sequencia), default=500)
    sequencia = inserir_espurios(sequencia, modo_cfg, tempo_max)

    # -----------------------------------------------------------------
    # Ordenar por timestamp
    # -----------------------------------------------------------------
    sequencia.sort(key=lambda x: x["timestamp_ms"])

    # -----------------------------------------------------------------
    # Extrair features e gerar label
    # -----------------------------------------------------------------
    features = extrair_features(sequencia, posicao, servico)
    label    = f"{nome_cenario}__{nome_posicao}__{modo}"

    return sequencia, features, label


# =============================================================================
# EXTRAÇÃO DE FEATURES PARA O LGBM
# =============================================================================

def extrair_features(sequencia, posicao, servico):
    """
    Extrai features estáticas da sequência de alarmes.

    Inclui features semânticas OTN que refletem o conhecimento de domínio:
    - Contagens por tipo de alarme
    - Contagens por camada OTN
    - Características da propagação (tem RDI? tem BDI? propagou downstream?)
    - Relação terminação vs pass-through nos nós afetados
    """
    if not sequencia:
        return {}

    # Contagem por tipo
    contagens = {
        f"qtd_{t.replace('-','_')}": sum(1 for a in sequencia if a["tipo"] == t)
        for t in TIPOS_ALARME
    }

    # Contagem por camada
    por_camada = {
        f"qtd_camada_{c}": sum(1 for a in sequencia if a["camada"] == c)
        for c in ["OTS", "OMS", "OCh", "OTU", "ODU", "NE"]
    }

    timestamps = [a["timestamp_ms"] for a in sequencia]
    total      = len(sequencia)
    duracao    = max(timestamps) - min(timestamps)

    nos_unicos     = list(set(a["no"] for a in sequencia))
    camadas_seq    = [a["camada"] for a in sequencia]
    camada_mais    = max(set(camadas_seq), key=camadas_seq.count)
    severidades    = [a["severidade"] for a in sequencia]
    mapa_sev       = {"Critical": 3, "Major": 2, "Minor": 1}
    sev_max        = max(severidades, key=lambda s: mapa_sev.get(s, 0))
    near_end       = sum(1 for a in sequencia if a["direcao"] == "near-end")
    prop_near      = round(near_end / total, 3) if total else 0

    tem_rdi        = int(any("RDI" in a["tipo"] or "BDI" in a["tipo"]
                              for a in sequencia))
    tem_bdi        = int(any("BDI" in a["tipo"] for a in sequencia))

    nos_com_ais    = set(a["no"] for a in sequencia if "AIS" in a["tipo"])
    prop_downstream = int(len(nos_com_ais) > 1)

    # Quantos nós afetados são pass-through vs terminação no serviço principal
    terminacoes    = set(servico.get("terminacao_odu", []))
    pass_through   = set(servico.get("pass_through", []))
    nos_alarm_set  = set(nos_unicos)
    nos_term_alarm = len(nos_alarm_set & terminacoes)
    nos_pt_alarm   = len(nos_alarm_set & pass_through)

    return {
        **contagens,
        **por_camada,
        "duracao_ms":             duracao,
        "qtd_total_alarmes":      total,
        "qtd_nos_afetados":       len(nos_unicos),
        "camada_mais_afetada":    camada_mais,
        "severidade_maxima":      sev_max,
        "primeiro_alarme":        sequencia[0]["tipo"],
        "ultimo_alarme":          sequencia[-1]["tipo"],
        "no_primeiro_alarme":     sequencia[0]["no"],
        "prop_near_end":          prop_near,
        "tem_rdi_ou_bdi":         tem_rdi,
        "tem_bdi":                tem_bdi,
        "propagacao_downstream":  prop_downstream,
        "qtd_servicos_afetados":  len(posicao.get("servicos_afetados", [])),
        "nos_terminacao_alarmados": nos_term_alarm,
        "nos_passthrough_alarmados": nos_pt_alarm,
    }


# =============================================================================
# GERAÇÃO DO DATASET COMPLETO
# =============================================================================

def gerar_dataset(amostras_por_combinacao=100, verbose=True):
    """
    Gera o dataset completo para todos os cenários, posições e modos.

    Para cada combinação válida (cenário × posição × modo), gera N amostras.

    Returns:
        DataFrame pandas com o dataset completo
    """
    registros = []
    total_combinacoes = 0

    for nome_cenario, cenario in CENARIOS_FALHA.items():
        for nome_posicao, posicao in POSICOES_FALHA.items():

            # Verifica compatibilidade entre cenário e posição
            tipo_posicao    = posicao["tipo"]
            propagacao      = cenario["propagacao"]

            # Cenários locais não fazem sentido em enlaces inteiros
            if tipo_posicao == "enlace" and propagacao == "local":
                continue

            # Cenários downstream em nós: apenas para falha_transponder
            # e falha_amplificador que têm comportamento de nó
            if tipo_posicao == "no" and propagacao == "downstream":
                if nome_cenario not in ["falha_transponder",
                                        "falha_amplificador"]:
                    continue

            # Gera para os dois modos de correlação
            for modo in ["sem_correlacao", "com_correlacao"]:
                total_combinacoes += 1
                label = f"{nome_cenario}__{nome_posicao}__{modo}"

                if verbose:
                    print(f"  [{modo[:3]}] {amostras_por_combinacao}x → {label}")

                for i in range(amostras_por_combinacao):
                    try:
                        sequencia, features, label_gerado = gerar_amostra(
                            nome_cenario, nome_posicao, modo
                        )

                        registro = {
                            "amostra_id":    f"{label}_{i:04d}",
                            "cenario":       nome_cenario,
                            "posicao":       nome_posicao,
                            "modo":          modo,
                            "label":         label_gerado,
                            "qtd_alarmes":   len(sequencia),

                            # Sequências para o LSTM
                            "seq_tipos":     ",".join(a["tipo"]
                                                      for a in sequencia),
                            "seq_nos":       ",".join(a["no"]
                                                      for a in sequencia),
                            "seq_camadas":   ",".join(a["camada"]
                                                      for a in sequencia),
                            "seq_timestamps":  ",".join(
                                str(a["timestamp_ms"]) for a in sequencia
                            ),
                            "seq_direcoes":  ",".join(a["direcao"]
                                                      for a in sequencia),
                            # Features para o LGBM
                            **features,
                        }
                        registros.append(registro)

                    except Exception as e:
                        if verbose:
                            print(f"    AVISO: erro na amostra {i} — {e}")

    if verbose:
        print(f"\nTotal combinações: {total_combinacoes}")
        print(f"Total amostras   : {len(registros)}")

    return pd.DataFrame(registros)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulador de alarmes OTN v2 — G.798/G.709"
    )
    parser.add_argument("--amostras", type=int, default=100)
    parser.add_argument("--output",   type=str,
                        default="../output/dataset_otn_v2.csv")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 65)
    print("SIMULADOR DE ALARMES OTN v2 — G.798/G.709")
    print("=" * 65)
    print(f"Amostras/combinação : {args.amostras}")
    print(f"Modos               : sem_correlacao + com_correlacao")
    print(f"Semente             : {args.seed}")
    print(f"Saída               : {args.output}")
    print("=" * 65 + "\n")

    df = gerar_dataset(amostras_por_combinacao=args.amostras, verbose=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 65)
    print("DATASET GERADO")
    print("=" * 65)
    print(f"Arquivo : {args.output}")
    print(f"Linhas  : {len(df)}")
    print(f"Colunas : {len(df.columns)}")
    print(f"\nDistribuição por cenário e modo:")
    print(df.groupby(["cenario", "modo"]).size().to_string())
