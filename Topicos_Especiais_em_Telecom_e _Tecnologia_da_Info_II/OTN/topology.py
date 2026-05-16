"""
=============================================================================
TOPOLOGIA OTN — VERSÃO 2
Baseada em G.709 e G.798
=============================================================================

NOVIDADE: cada serviço agora declara explicitamente quais nós são
terminações ODU e quais são pass-through. O simulador usa essa
informação para atribuir os alarmes corretos a cada nó.

Topologia:

    NE1 ——[E1]—— NE2 ——[E2]—— NE3 ——[E3]—— NE4
                               |
                             [E4]
                               |
                              NE5

Serviços:
    S1: NE1 → NE2 → NE3 → NE4  (ODU4)
        terminações: NE1, NE4
        pass-through: NE2, NE3

    S2: NE1 → NE2               (ODU2)
        terminações: NE1, NE2
        pass-through: (nenhum)

    S3: NE3 → NE5               (ODU1)
        terminações: NE3, NE5
        pass-through: (nenhum)
=============================================================================
"""

# =============================================================================
# NÓS
# =============================================================================
NOS = {
    "NE1": {"tipo": "OTN_ROADM", "capacidade": "100G",
            "camadas_suportadas": ["OTU4", "ODU4", "ODU2", "ODU1", "ODU0"],
            "tem_amplificador": False},

    "NE2": {"tipo": "OTN_ROADM", "capacidade": "100G",
            "camadas_suportadas": ["OTU4", "ODU4", "ODU2", "ODU1", "ODU0"],
            "tem_amplificador": True},

    "NE3": {"tipo": "OTN_ROADM", "capacidade": "100G",
            "camadas_suportadas": ["OTU4", "ODU4", "ODU2", "ODU1", "ODU0"],
            "tem_amplificador": True},

    "NE4": {"tipo": "OTN_ROADM", "capacidade": "100G",
            "camadas_suportadas": ["OTU4", "ODU4", "ODU2", "ODU1", "ODU0"],
            "tem_amplificador": False},

    "NE5": {"tipo": "OTN_ROADM", "capacidade": "100G",
            "camadas_suportadas": ["OTU2", "ODU2", "ODU1", "ODU0"],
            "tem_amplificador": False},
}

# =============================================================================
# ENLACES
# =============================================================================
ENLACES = {
    "E1": {"origem": "NE1", "destino": "NE2", "comprimento_km": 80,
           "num_amplificadores": 1, "potencia_nominal_dbm": 0.0},

    "E2": {"origem": "NE2", "destino": "NE3", "comprimento_km": 120,
           "num_amplificadores": 2, "potencia_nominal_dbm": 0.0},

    "E3": {"origem": "NE3", "destino": "NE4", "comprimento_km": 100,
           "num_amplificadores": 1, "potencia_nominal_dbm": 0.0},

    "E4": {"origem": "NE3", "destino": "NE5", "comprimento_km": 60,
           "num_amplificadores": 1, "potencia_nominal_dbm": 0.0},
}

# =============================================================================
# SERVIÇOS — com papéis explícitos de terminação e pass-through
# =============================================================================
SERVICOS = {
    "S1": {
        "descricao": "Serviço NE1→NE4 via NE2 e NE3",
        "origem": "NE1",
        "destino": "NE4",
        "caminho": ["NE1", "NE2", "NE3", "NE4"],
        "enlaces": ["E1", "E2", "E3"],
        "tipo_odu": "ODU4",
        "wavelength_thz": 193.1,
        # Papéis dos nós neste serviço
        "terminacao_odu": ["NE1", "NE4"],   # terminam o ODU — geram ODU-AIS/RDI
        "pass_through":   ["NE2", "NE3"],   # apenas passam — geram OTS/OMS
    },
    "S2": {
        "descricao": "Serviço NE1→NE2",
        "origem": "NE1",
        "destino": "NE2",
        "caminho": ["NE1", "NE2"],
        "enlaces": ["E1"],
        "tipo_odu": "ODU2",
        "wavelength_thz": 193.2,
        "terminacao_odu": ["NE1", "NE2"],
        "pass_through":   [],
    },
    "S3": {
        "descricao": "Serviço NE3→NE5",
        "origem": "NE3",
        "destino": "NE5",
        "caminho": ["NE3", "NE5"],
        "enlaces": ["E4"],
        "tipo_odu": "ODU1",
        "wavelength_thz": 193.3,
        "terminacao_odu": ["NE3", "NE5"],
        "pass_through":   [],
    },
}

# =============================================================================
# POSIÇÕES DE FALHA
#
# Cada posição define:
#   - tipo: "enlace" ou "no"
#   - enlace: qual enlace está com falha (None para falha em nó)
#   - nos_afetados: nós diretamente impactados pela falha física
#   - servicos_afetados: serviços que passam pela posição de falha
#   - upstream_adjacente: nó imediatamente ANTES da falha (envia tráfego)
#   - downstream_adjacente: nó imediatamente APÓS a falha (recebe tráfego)
#
# Para falhas em nó (local), upstream e downstream são o próprio nó e
# seu par no serviço mais afetado.
# =============================================================================
POSICOES_FALHA = {

    # Falhas em enlaces
    "E1_NE1_NE2": {
        "tipo": "enlace",
        "enlace": "E1",
        "nos_afetados": ["NE1", "NE2"],
        "servicos_afetados": ["S1", "S2"],
        "upstream_adjacente": "NE1",
        "downstream_adjacente": "NE2",
    },
    "E2_NE2_NE3": {
        "tipo": "enlace",
        "enlace": "E2",
        "nos_afetados": ["NE2", "NE3"],
        "servicos_afetados": ["S1"],
        "upstream_adjacente": "NE2",
        "downstream_adjacente": "NE3",
    },
    "E3_NE3_NE4": {
        "tipo": "enlace",
        "enlace": "E3",
        "nos_afetados": ["NE3", "NE4"],
        "servicos_afetados": ["S1"],
        "upstream_adjacente": "NE3",
        "downstream_adjacente": "NE4",
    },
    "E4_NE3_NE5": {
        "tipo": "enlace",
        "enlace": "E4",
        "nos_afetados": ["NE3", "NE5"],
        "servicos_afetados": ["S3"],
        "upstream_adjacente": "NE3",
        "downstream_adjacente": "NE5",
    },

    # Falhas em nós específicos
    "NE1_local": {
        "tipo": "no",
        "enlace": None,
        "nos_afetados": ["NE1"],
        "servicos_afetados": ["S1", "S2"],
        "upstream_adjacente": "NE1",
        "downstream_adjacente": "NE1",
    },
    "NE2_local": {
        "tipo": "no",
        "enlace": None,
        "nos_afetados": ["NE2"],
        "servicos_afetados": ["S1", "S2"],
        "upstream_adjacente": "NE2",
        "downstream_adjacente": "NE2",
    },
    "NE3_local": {
        "tipo": "no",
        "enlace": None,
        "nos_afetados": ["NE3"],
        "servicos_afetados": ["S1", "S3"],
        "upstream_adjacente": "NE3",
        "downstream_adjacente": "NE3",
    },
    "NE4_local": {
        "tipo": "no",
        "enlace": None,
        "nos_afetados": ["NE4"],
        "servicos_afetados": ["S1"],
        "upstream_adjacente": "NE4",
        "downstream_adjacente": "NE4",
    },
    "NE5_local": {
        "tipo": "no",
        "enlace": None,
        "nos_afetados": ["NE5"],
        "servicos_afetados": ["S3"],
        "upstream_adjacente": "NE5",
        "downstream_adjacente": "NE5",
    },
}
