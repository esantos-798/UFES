"""
=============================================================================
LIVRO DE REGRAS DE ALARMES OTN — VERSÃO 2
Baseado em ITU-T G.798 e G.709
=============================================================================

NOVIDADES DESTA VERSÃO:
    1. BDI corretamente modelado no nó upstream adjacente à falha
    2. Dois modos de operação:
       - sem_correlacao: todos os defeitos e alarmes consequentes aparecem
       - com_correlacao: NMS suprime alarmes consequentes, exibe causa raiz
    3. Papéis dos nós explicitados:
       - "upstream_adjacente"  : nó imediatamente antes da falha
       - "downstream_adjacente": nó imediatamente após a falha
       - "terminacao_odu_dest" : nó que termina o ODU no lado destino
       - "terminacao_odu_orig" : nó que termina o ODU no lado origem
       - "local"               : para falhas em nó específico
       - "remoto"              : nó par na comunicação (falhas locais)

REFERÊNCIAS G.798:
    - Sec 6.2  : Defect correlation e consequent actions
    - Sec 6.4  : Performance monitoring
    - Tabela 6-1: Lista de defects e alarmes por camada
    - Tabela 6-2: Consequent actions (supressão com correlação)
=============================================================================
"""

# =============================================================================
# TIPOS DE ALARME
# =============================================================================
TIPOS_ALARME = {
 
    # =========================================================================
    # CAMADA OTS — Optical Transmission Section
    # Seções 6.2.1.1, 6.2.6.4, 6.2.6.5, 6.2.6.7
    # =========================================================================
 
    "OTS-LOS-P": {
        "nome":       "Loss of Signal - Payload",
        "camada":     "OTS",
        "severidade": "Critical",
        "secao_g798": "6.2.1.1",
        "descricao":  "Perda do sinal óptico payload monitorada pelo NOM na "
                      "OTS ME. Indica falha do transmissor OTSi ou ruptura "
                      "do caminho óptico no OTS_ME.",
    },
    "OTS-LOS-O": {
        "nome":       "Loss of Signal - Overhead (OSC)",
        "camada":     "OTS",
        "severidade": "Critical",
        "secao_g798": "6.2.1.3",
        "descricao":  "Perda do sinal OSC (overhead). Indica falha do "
                      "transmissor OSC ou ruptura do caminho óptico do OSC.",
    },
    "OTS-PMI": {
        "nome":       "Payload Missing Indication",
        "camada":     "OTS",
        "severidade": "Critical",
        "secao_g798": "6.2.6.7.1",
        "descricao":  "Indica que o payload está ausente na fonte da trilha "
                      "OTS. Suprime alarmes dLOS-P causados por ausência de "
                      "payload já na origem, evitando alarmes falsos no sink.",
    },
    "OTS-BDI-P": {
        "nome":       "Backward Defect Indication - Payload",
        "camada":     "OTS",
        "severidade": "Major",
        "secao_g798": "6.2.6.4.1",
        "descricao":  "Indicação retroativa de defeito no payload. Inserida "
                      "pela função source em resposta ao recebimento de FDI-P "
                      "ou à detecção de defeito no media element. Permite "
                      "supervisão single-ended da trilha OTS.",
    },
    "OTS-BDI-O": {
        "nome":       "Backward Defect Indication - Overhead",
        "camada":     "OTS",
        "severidade": "Major",
        "secao_g798": "6.2.6.5.1",
        "descricao":  "Indicação retroativa de defeito no overhead. Inserida "
                      "pela função source em resposta ao recebimento de FDI-O "
                      "ou a defeitos na ME server. Permite supervisão "
                      "single-ended da trilha OTS.",
    },
    "OTS-TIM": {
        "nome":       "Trail Trace Identifier Mismatch",
        "camada":     "OTS",
        "severidade": "Major",
        "secao_g798": "6.2.2.1",
        "descricao":  "Incompatibilidade do Trail Trace Identifier (TTI) na "
                      "camada OTS-O. Indica conexão errada ou "
                      "misconfiguration do caminho óptico.",
    },
    "OTS-TSF-P": {
        "nome":       "Trail Signal Fail - Payload",
        "camada":     "OTS",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Falha do sinal de trilha no payload OTS. Fault cause "
                      "derivada de dLOS-P ou dTIM. Dispara ações consequentes "
                      "nas camadas cliente (OMS).",
    },
    "OTS-TSF-O": {
        "nome":       "Trail Signal Fail - Overhead",
        "camada":     "OTS",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Falha do sinal de trilha no overhead OTS. Fault cause "
                      "derivada de dLOS-O ou dTIM. Dispara ações consequentes "
                      "no OSC e camadas dependentes.",
    },
    "OTS-CI-SSF": {
        "nome":       "Server Signal Fail (Connection Information)",
        "camada":     "OTS",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Indicação de falha do sinal server recebida via "
                      "connection information pela camada OTS. Causa "
                      "propagação de FDI para as camadas cliente OMS.",
    },
     "OTS-AcTI": {
        "nome":       "Accepted Trail Trace Identifier",
        "camada":     "OTS",
        "severidade": "Warning",
        "secao_g798": "8.6",
        "descricao":  "Indicação de alteração no RxTI ",
    },
    # =========================================================================
    # CAMADA OMS — Optical Multiplex Section
    # Seções 6.2.6.1, 6.2.6.2, 6.2.6.4, 6.2.6.5, 6.2.6.7
    # =========================================================================
 
    "OMS-LOS-P": {
        "nome":       "Loss of Signal - Payload",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.1.2",
        "descricao":  "Perda do sinal óptico payload na camada OMS. "
                      "Monitorada nas funções de adaptação OTSi/OTSiG → OTU. "
                      "Indica falha do transmissor OTSi ou ruptura do caminho "
                      "óptico OTSi.",
    },
    "OMS-PMI": {
        "nome":       "Payload Missing Indication",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.6.7.1",
        "descricao":  "Análogo ao OTS-PMI, mas na camada OMS. Suprime "
                      "alarmes dLOS-P no sink quando o payload já está "
                      "ausente na source da trilha OMS.",
    },
    "OMS-FDI-P": {
        "nome":       "Forward Defect Indication - Payload",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.6.1.1",
        "descricao":  "Sinal de manutenção inserido no overhead não-associado "
                      "da camada OMS em resposta a defeitos no media element. "
                      "Suprime alarmes downstream nas camadas cliente (OCh) "
                      "causados por defeitos upstream detectados pelo server.",
    },
    "OMS-FDI-O": {
        "nome":       "Forward Defect Indication - Overhead",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.6.2.1",
        "descricao":  "Sinal de manutenção inserido pela OTS-O termination "
                      "sink em resposta a falhas OTS-O. Suprime alarmes no "
                      "OSC downstream causados por defeitos server.",
    },
    "OMS-BDI-P": {
        "nome":       "Backward Defect Indication - Payload",
        "camada":     "OMS",
        "severidade": "Major",
        "secao_g798": "6.2.6.4.1",
        "descricao":  "Indicação retroativa de defeito no payload OMS. "
                      "Inserida pela source em resposta a FDI-P ou defeito "
                      "na ME. Permite supervisão single-ended da trilha OMS.",
    },
    "OMS-BDI-O": {
        "nome":       "Backward Defect Indication - Overhead",
        "camada":     "OMS",
        "severidade": "Major",
        "secao_g798": "6.2.6.5.1",
        "descricao":  "Indicação retroativa de defeito no overhead OMS. "
                      "Inserida pela source em resposta a FDI-O ou defeitos "
                      "no server. Permite supervisão single-ended da trilha.",
    },
    "OMS-CI-SSF": {
        "nome":       "Server Signal Fail (Connection Information)",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Falha do sinal server recebida via CI pela camada OMS. "
                      "Normalmente propagada a partir de OTS-TSF. Causa "
                      "propagação de FDI para camadas cliente OCh/OTU.",
    },
    "OMS-TSF-P": {
        "nome":       "Trail Signal Fail - Payload",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Fault cause na camada OMS para o payload. Derivada de "
                      "dLOS-P, dFDI-P ou dTIM. Propaga CI_SSF para a camada "
                      "cliente OCh e insere FDI-P no overhead.",
    },
    "OMS-TSF-O": {
        "nome":       "Trail Signal Fail - Overhead",
        "camada":     "OMS",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Fault cause na camada OMS para o overhead/OSC. "
                      "Derivada de dFDI-O ou dTIM. Propaga CI_SSF para "
                      "camadas dependentes do OSC e insere FDI-O.",
    },

 
    # =========================================================================
    # CAMADA OCh — Optical Channel
    # Seção 6.2.6.8.1 (OCI), 6.2.6.1.1 (FDI-P), 6.2.6.2.1 (FDI-O)
    # =========================================================================
 
    "OCh-LOS": {
        "nome":       "Optical Channel Loss of Signal",
        "camada":     "OCh",
        "severidade": "Critical",
        "secao_g798": "6.2.1.x",
        "descricao":  "Perda do canal óptico específico. Detectado na "
                      "terminação OCh-O sink quando o canal não está "
                      "presente ou está abaixo do limiar de potência.",
    },
    "OCh-OCI": {
        "nome":       "Open Connection Indication",
        "camada":     "OCh",
        "severidade": "Major",
        "secao_g798": "6.2.6.8.1",
        "descricao":  "Indica que o ponto de conexão de saída (OCh_CP) não "
                      "está conectado a um ponto de entrada. Declarado quando "
                      "a função de conexão OCh recebe comando de desconexão.",
    },
    "OCh-FDI-P": {
        "nome":       "Forward Defect Indication - Payload",
        "camada":     "OCh",
        "severidade": "Critical",
        "secao_g798": "6.2.6.1.1",
        "descricao":  "FDI-P monitorada na camada OCh-O. Suprime alarmes "
                      "downstream causados por defeitos upstream no OMS.",
    },
 
    # =========================================================================
    # CAMADA OTU — Optical Transport Unit
    # Seções 6.2.5.1 (LOF), 6.2.5.3 (LOFLOM), 6.2.3.4 (DEG),
    #         6.2.6.3.1 (AIS), 6.2.6.6.1 (BDI), 6.2.6.10.1 (IAE),
    #         6.2.6.11.1 (BIAE), 6.2.2.1 (TIM)
    # =========================================================================
 
    "OTU-LOF": {
        "nome":       "Loss of Frame",
        "camada":     "OTU",
        "severidade": "Critical",
        "secao_g798": "6.2.5.1",
        "descricao":  "Perda de alinhamento de frame OTU. Declarado quando "
                      "o processo de frame alignment permanece no estado "
                      "OOF por 3 ms. Detectado no nó de terminação OTU.",
    },
    "OTU-LOM": {
        "nome":       "Loss of Multiframe",
        "camada":     "OTU",
        "severidade": "Critical",
        "secao_g798": "6.2.5.2",
        "descricao":  "Perda de alinhamento de multiframe OTU. Declarado "
                      "quando o processo de multiframe alignment permanece "
                      "no estado OOM por 3 ms.",
    },
    "OTU-LOFLOM": {
        "nome":       "Loss of Frame and Multiframe",
        "camada":     "OTU",
        "severidade": "Critical",
        "secao_g798": "6.2.5.3",
        "descricao":  "Perda simultânea de alinhamento de frame e multiframe "
                      "OTU. Declarado quando o processo conjunto permanece "
                      "no estado OOF por 3 ms.",
    },
    "OTU-AIS": {
        "nome":       "Alarm Indication Signal",
        "camada":     "OTU",
        "severidade": "Critical",
        "secao_g798": "6.2.6.3.1",
        "descricao":  "Sinal de indicação de alarme na camada OTUk. "
                      "Detectado pelo padrão CBR generic AIS (PN-11 reverso). "
                      "Nota G.798: OTN deve detectar mas não é obrigado "
                      "a gerar OTUk-AIS exceto para suporte a futuro server.",
    },
    "OTU-SSF": {
        "nome":       "Server Signal Fail",
        "camada":     "OTU",
        "severidade": "Critical",
        "secao_g798": "6.2.6.3.1",
        "descricao":  "Falha no sinal da camada servidora reportada fora do "
                      "overhead no sentido adaptação para terminação ",
    },
    "OTU-BDI": {
        "nome":       "Backward Defect Indication",
        "camada":     "OTU",
        "severidade": "Major",
        "secao_g798": "6.2.6.6.1",
        "descricao":  "Indicação retroativa de defeito na camada OTU. "
                      "Declarado quando o bit BDI no campo SM overhead "
                      "(byte 3, bit 5) é '1' por X frames consecutivos "
                      "(X=5 para OTUk). Enviado upstream pelo nó que "
                      "detectou a falha.",
    },
    "OTU-DEG": {
        "nome":       "Signal Degrade",
        "camada":     "OTU",
        "severidade": "Major",
        "secao_g798": "6.2.3.4",
        "descricao":  "Degradação do sinal OTU. BER acima do limiar de dDEG. "
                      "Monitorado pelo algoritmo de contagem de errored blocks "
                      "por segundo. EBC descontado se dIAE ativo no período.",
    },
    "OTU-TSD": {
        "nome":       "Traisl Signal Degraded",
        "camada":     "OTU",
        "severidade": "Major",
        "secao_g798": "6.2.3.4",
        "descricao":  "Degradação do sinal OTU.",
    },    
    "OTU-TIM": {
        "nome":       "Trail Trace Identifier Mismatch",
        "camada":     "OTU",
        "severidade": "Major",
        "secao_g798": "6.2.2.1",
        "descricao":  "Incompatibilidade do TTI na camada OTU (campo SM). "
                      "Detectado por comparação SAPI/DAPI do RxTI com "
                      "MI_ExSAPI/DAPI. Indica cross-connection errada.",
    },
    "OTU-IAE": {
        "nome":       "Incoming Alignment Error",
        "camada":     "OTU",
        "severidade": "Minor",
        "secao_g798": "6.2.6.10.1",
        "descricao":  "Erro de alinhamento de entrada na camada OTUk. "
                      "Declarado quando bit IAE no campo SM (byte 3, bit 6) "
                      "é '1' por 5 frames consecutivos. Suprime PM data "
                      "(EBC e DS) — não gera fault cause.",
    },
    "OTU-BIAE": {
        "nome":       "Backward Incoming Alignment Error",
        "camada":     "OTU",
        "severidade": "Minor",
        "secao_g798": "6.2.6.11.1",
        "descricao":  "Erro de alinhamento de entrada retroativo OTU. "
                      "Declarado quando bits BEI/BIAE no campo SM são '1011' "
                      "por 3 frames consecutivos. Suprime PM far-end — "
                      "não gera fault cause.",
    },
    "OTU-TSF": {
        "nome":       "Trail Signal Fail",
        "camada":     "OTU",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Fault cause na camada OTU. Derivada de dLOF, dLOM, "
                      "dAIS ou dTIM. Propaga CI_SSF para a camada cliente ODU "
                      "e inicia ações consequentes.",
    },
    "OTU-N_EBC": {
        "nome":       "Near-end Errored Block Count",
        "camada":     "OTU",
        "severidade": "Minor",
        "secao_g798": "N/A",
        "descricao":  "Capacidade de correção FEC sendo consumida acima do "
                      "limiar na estação local. Indica degradação de OSNR "
                      " antes que o sinal atinja o limiar de dDEG.",
    },
    "OTU-F_EBC": {
        "nome":       "Far-end Errored Block Count",
        "camada":     "OTU",
        "severidade": "Minor",
        "secao_g798": "N/A",
        "descricao":  "Capacidade de correção FEC sendo consumida acima do "
                      "limiar na estação remota. Indica degradação de OSNR "
                      " antes que o sinal atinja o limiar de dDEG.",
    },
    "OTU-MSIM": {
        "nome":       "Multiplex Structure Identifier Mismatch",
        "camada":     "OTU",
        "severidade": "Major",
        "secao_g798": "6.2.9.1",
        "descricao":  "Incompatibilidade do identificador de estrutura "
                      "multiplex na camada OTUCn. AcMSI diferente de ExMSI "
                      "em um ou mais tributary slots.",
    },
    "OTU-AcTI": {
        "nome":       "Accepted Trail Trace Identifier",
        "camada":     "OTU",
        "severidade": "Warning",
        "secao_g798": "8.6",
        "descricao":  "Indicação de alteração no RxTI ",
    },
    "OTU-AcMSI": {
        "nome":       "Accepted Multiplex Structure Identifier",
        "camada":     "OTU",
        "severidade": "Warning",
        "secao_g798": "8.7.2.2",
        "descricao":  "Indicação de alteração no RxMSI ",
    },  
    # =========================================================================
    # CAMADA ODU — Optical Data Unit (Path Monitor e Tandem Connection)
    # Seções 6.2.6.3.2 (AIS), 6.2.6.6.1 (BDI), 6.2.6.8.2 (OCI),
    #         6.2.6.9.1 (LCK), 6.2.1.5.1 (LTC), 6.2.3.4 (DEG),
    #         6.2.2.1 (TIM), 6.2.4.1 (PLM), 6.2.9.1 (MSIM)
    # =========================================================================
 
    "ODU-AIS": {
        "nome":       "Alarm Indication Signal",
        "camada":     "ODU",
        "severidade": "Critical",
        "secao_g798": "6.2.6.3.2",
        "descricao":  "Sinal de indicação de alarme na camada ODU. "
                      "Declarado quando AcSTAT = '111'. Gerado APENAS no "
                      "nó de terminação ODU — nunca em pass-through. "
                      "Consequente de OTU-TSF ou CI_SSF.",
    },
    "ODU-BDI": {
        "nome":       "Backward Defect Indication",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.6.6.1",
        "descricao":  "Indicação de defeito remoto na camada ODU (PM). "
                      "Declarado quando bit BDI no campo PM overhead "
                      "(byte 3, bit 5) é '1' por 5 frames consecutivos. "
                      "Enviado de volta ao nó de terminação ODU origem.",
    },
    "ODU-OCI": {
        "nome":       "Open Connection Indication",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.6.8.2",
        "descricao":  "Conexão aberta na camada ODU. Declarado quando "
                      "AcSTAT = '110'. Indica que o tributary slot "
                      "não está conectado a uma fonte de sinal.",
    },
    "ODU-LCK": {
        "nome":       "Locked",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.6.9.1",
        "descricao":  "ODU travado administrativamente. Declarado quando "
                      "AcSTAT = '101'. Indica que o administrador bloqueou "
                      "o ODU intencionalmente — não é falha de rede.",
    },
    "ODU-LTC": {
        "nome":       "Loss of Tandem Connection",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.1.5.1",
        "descricao":  "Perda da conexão tandem (TCM). Declarado quando "
                      "AcSTAT = '000' no campo TCMi. Indica perda da "
                      "supervisão do segmento de tandem connection.",
    },
    "ODU-DEG": {
        "nome":       "Signal Degrade",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.3.4",
        "descricao":  "Degradação do sinal ODU. BER acima do limiar de dDEG "
                      "monitorado pelo algoritmo de errored blocks. "
                      "EBC descontado nos períodos com dIAE ativo.",
    },
    "OTU-TSD": {
        "nome":       "Trail Signal Degraded",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.3.4",
        "descricao":  "Degradação do sinal ODU.",
    },       
    "ODU-TIM": {
        "nome":       "Trail Trace Identifier Mismatch",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.2.1",
        "descricao":  "Incompatibilidade do TTI na camada ODU (campo PM). "
                      "Indica que o ODU foi roteado incorretamente ou "
                      "há misconfiguration de cross-connection.",
    },
    "ODU-PLM": {
        "nome":       "Payload Label Mismatch",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.4.1",
        "descricao":  "Incompatibilidade do tipo de payload na camada ODUP. "
                      "AcPT diferente do ExPT. Indica mismatch entre o tipo "
                      "de cliente esperado e o recebido no OPU.",
    },
    "ODU-MSIM": {
        "nome":       "Multiplex Structure Identifier Mismatch",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.9.1",
        "descricao":  "Incompatibilidade do identificador de estrutura "
                      "multiplex na camada ODUkP. AcMSI diferente de ExMSI "
                      "em um ou mais tributary slots.",
    },
    "ODU-IAE": {
        "nome":       "Incoming Alignment Error",
        "camada":     "ODU",
        "severidade": "Minor",
        "secao_g798": "6.2.6.10.2",
        "descricao":  "Erro de alinhamento de entrada na camada ODUT. "
                      "Declarado quando AcSTAT = '010'. Suprime PM data "
                      "incorretos causados por frame slips — não gera "
                      "fault cause.",
    },
    "ODU-BIAE": {
        "nome":       "Backward Incoming Alignment Error",
        "camada":     "ODU",
        "severidade": "Minor",
        "secao_g798": "6.2.6.11.1",
        "descricao":  "Erro de alinhamento de entrada retroativo na camada "
                      "ODUT. Declarado quando bits BEI/BIAE = '1011' por "
                      "3 frames consecutivos. Suprime PM far-end incorretos.",
    },
    "ODU-TSF": {
        "nome":       "Trail Signal Fail",
        "camada":     "ODU",
        "severidade": "Critical",
        "secao_g798": "6.2.6.3.1",
        "descricao":  "Falha no sinal da camada servidora reportada fora do "
                      "overhead no sentido terminação para adaptação ",
    },     
    "ODU-SSF": {
        "nome":       "Server Signal Fail (Connection Information)",
        "camada":     "ODU",
        "severidade": "Critical",
        "secao_g798": "6.2.x",
        "descricao":  "Falha do sinal server recebida via CI pela camada ODU. "
                      "Propagada a partir de OTU-TSF. Dispara geração de "
                      "ODU-AIS no tributary slot correspondente.",
    },
    "ODU-CSF": {
        "nome":       "Client Signal Fail",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.10",
        "descricao":  "Falha do sinal cliente encapsulado no OPU. Declarado "
                      "quando bit CSF no PSI[2] é '1' por X multiframes "
                      "consecutivos de 256 frames. Indica falha na camada "
                      "cliente (ex: Ethernet, SDH) transportada pelo OTN.",
    },
    "ODU-FOP-PM": {
        "nome":       "Failure of Protocol - Provisioning Mismatch",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.7.1.1",
        "descricao":  "Falha de protocolo de proteção linear ODU por "
                      "incompatibilidade de provisionamento. Declarado "
                      "quando o bit B do protocolo APS transmitido e aceito "
                      "não coincidem. Ref: G.873.1.",
    },
    "ODU-FOP-NR": {
        "nome":       "Failure of Protocol - No Response",
        "camada":     "ODU",
        "severidade": "Major",
        "secao_g798": "6.2.7.1.2",
        "descricao":  "Falha de protocolo de proteção linear ODU por falta "
                      "de resposta. Declarado quando o requested signal e "
                      "o bridge signal no protocolo APS não coincidem em 1s. "
                      "Ref: G.873.1.",
    },
     "ODU-N_EBC": {
        "nome":       "Near-end Errored Block Count",
        "camada":     "OTU",
        "severidade": "Minor",
        "secao_g798": "N/A",
        "descricao":  "Capacidade de correção FEC sendo consumida acima do "
                      "limiar na estação local. Indica degradação de OSNR "
                      " antes que o sinal atinja o limiar de dDEG.",
    },
    "ODU-F_EBC": {
        "nome":       "Far-end Errored Block Count",
        "camada":     "OTU",
        "severidade": "Minor",
        "secao_g798": "N/A",
        "descricao":  "Capacidade de correção FEC sendo consumida acima do "
                      "limiar na estação remota. Indica degradação de OSNR "
                      " antes que o sinal atinja o limiar de dDEG.",
    },
    "ODU-AcTI": {
        "nome":       "Accepted Trail Trace Identifier",
        "camada":     "ODU",
        "severidade": "Warning",
        "secao_g798": "8.6",
        "descricao":  "Indicação de alteração no RxTI ",
    },
    "ODU-AcMSI": {
        "nome":       "Accepted Multiplex Structure Identifier",
        "camada":     "ODU",
        "severidade": "Warning",
        "secao_g798": "8.7.2.2",
        "descricao":  "Indicação de alteração no RxMSI ",
    },   
    # =========================================================================
    # CAMADA OPU — Optical Payload Unit (via ODU)
    # Seção 6.2.5.3 (LOFLOM — alinhamento multiplex OPU)
    # =========================================================================
 
    "OPU-LOFLOM": {
        "nome":       "Loss of Frame and Multiframe (OPU Multiplex)",
        "camada":     "OPU",
        "severidade": "Critical",
        "secao_g798": "6.2.5.3",
        "descricao":  "Perda de alinhamento de frame e multiframe na "
                      "estrutura multiplex OPU. Declarado quando o processo "
                      "de alinhamento conjunto permanece em OOF por 3 ms. "
                      "Relevante para ODU multiplexado.",
    },
    "OPU-PLM": {
        "nome":       "Payload Label Mismatch",
        "camada":     "OPU",
        "severidade": "Major",
        "secao_g798": "6.2.4.1",
        "descricao":  "Mismatch do payload type no campo PT do PSI OPU. "
                      "Indica incompatibilidade entre o cliente esperado e "
                      "o sinal recebido no tributary. Detectado na ODUP.",
    },
    "OPU-AcPT": {
        "nome":       "Accepted Payload Type",
        "camada":     "ODU",
        "severidade": "Warning",
        "secao_g798": "6.2.4.1",
        "descricao":  "Indicação de alteração no RxPT ",
    },   
    "OPU-AcPTI": {
        "nome":       "Accepted Payload Type Identifier",
        "camada":     "ODU",
        "severidade": "Warning",
        "secao_g798": "6.2.4.x",
        "descricao":  "Indicação de alteração no RxPTI ",
    },     
    # =========================================================================
    # CAMADA NE — Network Element (equipamento)
    # Alarmes de equipamento — não definidos diretamente pelo G.798
    # mas essenciais para o contexto de gerenciamento OTN
    # =========================================================================
 
    "NE-EQPT-FAIL": {
        "nome":       "Equipment Failure",
        "camada":     "NE",
        "severidade": "Critical",
        "secao_g798": "N/A",
        "descricao":  "Falha interna de equipamento — placa de linha, módulo "
                      "óptico SFP/CFP, DSP ou componente de hardware. "
                      "Detectado pelo sistema de auto-diagnóstico do NE. "
                      "Causa raiz para OTU-LOF e ODU-AIS consequentes.",
    },
    "NE-PWR-FAIL": {
        "nome":       "Power Failure",
        "camada":     "NE",
        "severidade": "Critical",
        "secao_g798": "N/A",
        "descricao":  "Falha de alimentação elétrica do equipamento ou "
                      "de uma unidade de potência (PSU). Pode afetar "
                      "múltiplos serviços simultaneamente.",
    },
    "NE-TEMP-HIGH": {
        "nome":       "High Temperature Alarm",
        "camada":     "NE",
        "severidade": "Major",
        "secao_g798": "N/A",
        "descricao":  "Temperatura interna do equipamento acima do limiar "
                      "de alarme. Pode preceder falha de equipamento se não "
                      "tratado. Monitorado pelo sistema de cooling do NE.",
    },
}

# =============================================================================
# CENÁRIOS DE FALHA — VERSÃO 2
# =============================================================================
CENARIOS_FALHA = {

    # -------------------------------------------------------------------------
    # CENÁRIO 1: Corte de fibra
    # -------------------------------------------------------------------------
    "corte_fibra": {
        "descricao": "Corte físico da fibra óptica em um enlace",
        "causa_raiz": "Interrupção total do sinal óptico por dano físico",
        "camadas_afetadas": ["OTS", "OMS", "OCh", "OTU", "ODU"],
        "propagacao": "downstream",

        "sem_correlacao": {
            "alarmes": [
                {"tipo": "OTS-LOS-P",      "delta_ms": 0,   "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 50,
                 "logica": "Downstream adjacente perde sinal — causa raiz"},
                {"tipo": "OTU-BDI", "delta_ms": 80,  "papel_no": "downstream_adjacente", "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "Downstream envia BDI upstream — G.798 Sec 6.2.3"},
                {"tipo": "OTU-TSF",  "delta_ms": 100, "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "MS-AIS propagado downstream a partir do LOS"},
                {"tipo": "OTU-LOF", "delta_ms": 150, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 70,
                 "logica": "Terminação ODU destino perde frame OTU"},
                {"tipo": "OTU-AIS", "delta_ms": 180, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "OTU-AIS consequente do OTU-LOF"},
                {"tipo": "ODU-AIS", "delta_ms": 210, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 70,
                 "logica": "ODU-AIS APENAS na terminação — não em pass-through"},
                {"tipo": "ODU-BDI", "delta_ms": 260, "papel_no": "terminacao_odu_orig",  "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "ODU-BDI enviado de volta à terminação origem"},
            ],
            "alarmes_possiveis_espurios": ["SD", "BBE-high"],
            "probabilidade_espurio": 0.3,
        },

        "com_correlacao": {
            "alarmes": [
                {"tipo": "OTS-LOS-P",     "delta_ms": 0,   "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 50,
                 "logica": "Causa raiz — sempre exibida"},
                {"tipo": "OTU-BDI","delta_ms": 80,  "papel_no": "downstream_adjacente", "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "BDI sobrevive — indica nó que detectou a falha"},
                {"tipo": "ODU-SSF","delta_ms": 210, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": False, "jitter_ms": 100,
                 "logica": "ODU-SSF pode sobreviver dependendo do fabricante"},
                {"tipo": "ODU-BDI","delta_ms": 260, "papel_no": "terminacao_odu_orig",  "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "ODU-BDI sobrevive — confirma percepção da terminação destino"},
            ],
            "alarmes_possiveis_espurios": ["SD"],
            "probabilidade_espurio": 0.15,
        },
    },

    # -------------------------------------------------------------------------
    # CENÁRIO 2: Falha de amplificador
    # -------------------------------------------------------------------------
    "falha_amplificador": {
        "descricao": "Falha ou degradação severa de amplificador EDFA",
        "causa_raiz": "Amplificador sem ganho ou sem bombeamento",
        "camadas_afetadas": ["OTS", "OMS", "OTU", "ODU"],
        "propagacao": "downstream",

        "sem_correlacao": {
            "alarmes": [
                {"tipo": "OTS-LOS-P",   "delta_ms": 0,   "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 40,
                 "logica": "Downstream detecta perda de potência"},
                {"tipo": "OTU-BDI", "delta_ms": 80,  "papel_no": "downstream_adjacente", "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "Downstream envia BDI upstream — G.798 Sec 6.2.3"},
                {"tipo": "OTU-TSF",  "delta_ms": 100, "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "MS-AIS propagado downstream a partir do LOS"},
                {"tipo": "OTU-LOF", "delta_ms": 150, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 70,
                 "logica": "Terminação ODU destino perde frame OTU"},
                {"tipo": "OTU-AIS", "delta_ms": 180, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "OTU-AIS consequente do OTU-LOF"},
                {"tipo": "ODU-AIS", "delta_ms": 210, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 70,
                 "logica": "ODU-AIS APENAS na terminação — não em pass-through"},
                {"tipo": "ODU-BDI", "delta_ms": 260, "papel_no": "terminacao_odu_orig",  "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "ODU-BDI enviado de volta à terminação origem"},
            ],
            "alarmes_possiveis_espurios": ["SD", "FEC-deg"],
            "probabilidade_espurio": 0.25,
        },

        "com_correlacao": {
            "alarmes": [
                {"tipo": "OTS-LOS-P",     "delta_ms": 0,   "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 50,
                 "logica": "Causa raiz — sempre exibida"},
                {"tipo": "OTU-BDI","delta_ms": 80,  "papel_no": "downstream_adjacente", "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "BDI sobrevive — indica nó que detectou a falha"},
                {"tipo": "ODU-SSF","delta_ms": 210, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": False, "jitter_ms": 100,
                 "logica": "ODU-SSF pode sobreviver dependendo do fabricante"},
                {"tipo": "ODU-BDI","delta_ms": 260, "papel_no": "terminacao_odu_orig",  "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "ODU-BDI sobrevive — confirma percepção da terminação destino"},
            ],
            "alarmes_possiveis_espurios": ["FEC-deg"],
            "probabilidade_espurio": 0.15,
        },
    },

    # -------------------------------------------------------------------------
    # CENÁRIO 3: Degradação de OSNR
    # -------------------------------------------------------------------------
    #"degradacao_osnr": {
    #    "descricao": "Degradação gradual do OSNR por acúmulo de ruído ASE",
    #    "causa_raiz": "OSNR abaixo do limiar por acúmulo de ASE",
    #    "camadas_afetadas": ["OMS", "OTU", "ODU"],
    #    "propagacao": "local",

    #    "sem_correlacao": {
    #        "alarmes": [
    #            {"tipo": "SD",       "delta_ms": 0,    "papel_no": "terminacao_odu_dest", "direcao": "far-end", "obrigatorio": True,  "jitter_ms": 200,
    #             "logica": "SD detectado na terminação destino"},
    #            {"tipo": "FEC-deg",  "delta_ms": 500,  "papel_no": "terminacao_odu_dest", "direcao": "far-end", "obrigatorio": True,  "jitter_ms": 300,
    #             "logica": "FEC sendo consumido"},
    #            {"tipo": "BBE-high", "delta_ms": 800,  "papel_no": "terminacao_odu_dest", "direcao": "far-end", "obrigatorio": False, "jitter_ms": 400,
    #             "logica": "BBE acima do limiar"},
    #            {"tipo": "ES-high",  "delta_ms": 1200, "papel_no": "terminacao_odu_dest", "direcao": "far-end", "obrigatorio": False, "jitter_ms": 500,
    #             "logica": "Errored seconds acima do limiar"},
    #        ],
    #        "alarmes_possiveis_espurios": ["MS-DEG"],
    #        "probabilidade_espurio": 0.2,
    #    },

    #    "com_correlacao": {
    #        "alarmes": [
    #            {"tipo": "SD",      "delta_ms": 0,   "papel_no": "terminacao_odu_dest", "direcao": "far-end", "obrigatorio": True, "jitter_ms": 200,
    #             "logica": "SD — causa raiz visível"},
    #            {"tipo": "FEC-deg", "delta_ms": 500, "papel_no": "terminacao_odu_dest", "direcao": "far-end", "obrigatorio": True, "jitter_ms": 300,
    #             "logica": "FEC-deg sobrevive"},
    #        ],
    #        "alarmes_possiveis_espurios": [],
    #        "probabilidade_espurio": 0.1,
    #    },
    #},

    # -------------------------------------------------------------------------
    # CENÁRIO 4: Filtro desalinhado
    # -------------------------------------------------------------------------
    #"filtro_desalinhado": {
    #    "descricao": "Desalinhamento ou configuração incorreta de filtro WSS",
    #    "causa_raiz": "Filtro WSS com desvio de frequência ou passband estreito",
    #    "camadas_afetadas": ["OMS", "OCh", "OTU"],
    #    "propagacao": "local",

    #    "sem_correlacao": {
    #        "alarmes": [
    #            {"tipo": "SD",      "delta_ms": 0,   "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 150,
    #             "logica": "SD na terminação destino"},
    #            {"tipo": "OCh-LOS", "delta_ms": 300, "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": False, "jitter_ms": 200,
    #             "logica": "OCh-LOS no nó adjacente ao filtro"},
    #            {"tipo": "FEC-deg", "delta_ms": 600, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 250,
    #             "logica": "FEC-deg consequente"},
    #            {"tipo": "BBE-high","delta_ms": 900, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": False, "jitter_ms": 300,
    #             "logica": "BBE-high consequente"},
    #        ],
    #        "alarmes_possiveis_espurios": ["ES-high", "SD"],
    #        "probabilidade_espurio": 0.2,
    #    },

    #    "com_correlacao": {
    #       "alarmes": [
    #            {"tipo": "SD",      "delta_ms": 0,   "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 150,
    #             "logica": "SD — causa raiz"},
    #            {"tipo": "OCh-LOS", "delta_ms": 300, "papel_no": "downstream_adjacente", "direcao": "near-end", "obrigatorio": False, "jitter_ms": 200,
    #             "logica": "OCh-LOS pode sobreviver — indica nó com filtro problemático"},
    #            {"tipo": "FEC-deg", "delta_ms": 600, "papel_no": "terminacao_odu_dest",  "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 250,
    #             "logica": "FEC-deg sobrevive"},
    #        ],
    #        "alarmes_possiveis_espurios": [],
    #        "probabilidade_espurio": 0.1,
    #    },
    #},

    # -------------------------------------------------------------------------
    # CENÁRIO 5: Falha de transponder
    # -------------------------------------------------------------------------
    "falha_transponder": {
        "descricao": "Falha de placa transponder ou módulo óptico",
        "causa_raiz": "Defeito interno no transponder — laser, DSP ou módulo SFP",
        "camadas_afetadas": ["NE", "OTU", "ODU"],
        "propagacao": "local",

        "sem_correlacao": {
            "alarmes": [
                {"tipo": "EQPT-fail","delta_ms": 0,   "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 30,
                 "logica": "Falha de equipamento na terminação origem"},
                {"tipo": "OTU-LOF",  "delta_ms": 80,  "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 50,
                 "logica": "LOF consequente no mesmo nó"},
                {"tipo": "OTU-AIS",  "delta_ms": 120, "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "OTU-AIS consequente"},
                {"tipo": "ODU-AIS",  "delta_ms": 160, "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 60,
                 "logica": "ODU-AIS na terminação origem"},
                {"tipo": "OTU-BDI",  "delta_ms": 190, "papel_no": "terminacao_odu_dest", "direcao": "far-end",  "obrigatorio": True,  "jitter_ms": 70,
                 "logica": "Terminação destino detecta LOF e envia BDI de volta"},
                {"tipo": "ODU-BDI",  "delta_ms": 230, "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True,  "jitter_ms": 80,
                 "logica": "ODU-BDI de volta à terminação origem"},
            ],
            "alarmes_possiveis_espurios": ["LOS-P", "BBE-high"],
            "probabilidade_espurio": 0.15,
        },

        "com_correlacao": {
            "alarmes": [
                {"tipo": "EQPT-fail","delta_ms": 0,   "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True, "jitter_ms": 30,
                 "logica": "EQPT-fail — causa raiz"},
                {"tipo": "OTU-BDI",  "delta_ms": 190, "papel_no": "terminacao_odu_dest", "direcao": "far-end",  "obrigatorio": True, "jitter_ms": 70,
                 "logica": "OTU-BDI sobrevive — indica que destino percebeu"},
                {"tipo": "ODU-BDI",  "delta_ms": 230, "papel_no": "terminacao_odu_orig", "direcao": "near-end", "obrigatorio": True, "jitter_ms": 80,
                 "logica": "ODU-BDI sobrevive"},
            ],
            "alarmes_possiveis_espurios": [],
            "probabilidade_espurio": 0.1,
        },
    },
}

# =============================================================================
# ALARMES ESPÚRIOS GLOBAIS
# =============================================================================
ALARMES_ESPURIOS_GLOBAIS     = ["BBE-high", "ES-high", "SD", "PWR-fail"]
PROBABILIDADE_ESPURIO_GLOBAL = 0.1
