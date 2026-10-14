# VLC Test Bed

Automatiza experimentos do enlace 5G NR sobre VLC: downlink pelo ADALM2000 e uplink pelo ADALM-PLUTO ou por simulação.

- `main_start.m`: configura e inicia a varredura.
- `main_experimental.m`: executa uma combinação de parâmetros.
- `ofdm3D.m`: visualização OFDM.

## Parametrização

Todos os parâmetros de varredura são definidos no início de `main_start.m`; os parâmetros do enlace são consumidos por `main_experimental.m`.

| Grupo | Parâmetros | Descrição |
| --- | --- | --- |
| MCS | `DCItypes` | Vetor de índices MCS a serem testados. Cada índice é convertido em modulação e taxa de código por `getMCSLookupTable`. |
| Bancada óptica | `Idc`, `distance` | Corrente de polarização do emissor óptico e distância física do enlace; ambos identificam a condição do ensaio. |
| Canal | `downlinkCH`, `uplinkCH` | Seleciona `software` ou `adalm2000` no downlink e `software` ou `adalm-pluto` no uplink. |
| Critério de qualidade | `evmDLmin`, `evmDLmax` | Limites de EVM usados para avaliar o downlink na condição selecionada. |
| NR | `pSCS`, `pSizeGrid`, `pBW`, `pNCellID`, `pLayers` | Definidos em `main_experimental.m`; configuram a numerologia e a grade do gNB/UE. |
| Hardware | `pPlutoArg`, `pM2K`, IPs dos dispositivos | Define ganho/frequência RF, portadora VLC, taxas ADC/DAC e endereços da bancada. |

## Como rodar

Em `main_start.m`, configure `DCItypes`, `Idc`, `distance`, `downlinkCH` e `uplinkCH`; depois execute `main_start.m`.

Para teste sem equipamentos, use `downlinkCH = 'software'` e `uplinkCH = 'software'`. Para a bancada física, use `downlinkCH = 'adalm2000'` e `uplinkCH = 'adalm-pluto'`, revise os IPs de cada equipamento e confirme a calibração do ADALM2000.

Os resultados são propositalmente ignorados pelo Git.
