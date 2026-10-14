# Full-duplex / enlace integrado

`main_experimental.m` implementa o fluxo gNB→UE no downlink e UE→gNB no uplink, incluindo ACK e métricas de BER/EVM.

## Como rodar

Abra `main_experimental.m`. Primeiro use:

```matlab
pCHDownlink = 'software';
pCHUplink = 'software';
```

Para a bancada integrada, a configuração prevista é `pCHDownlink = 'adalm2000'` para VLC e `pCHUplink = 'adalm-pluto'` para RF.

## Parametrização

Os parâmetros são definidos no início de `main_experimental.m`.

| Grupo | Parâmetros | Descrição |
| --- | --- | --- |
| Sentido do enlace | `pCHDownlink`, `pCHUplink` | Seleciona o canal de cada sentido. Downlink: `software` ou `adalm2000`; uplink: `software` ou `adalm-pluto`. |
| Controle | `pMaxIt`, `pMaxRetry`, `pPlotFigures` | Define número máximo de iterações, tentativas de retransmissão e habilita/desabilita gráficos. |
| gNB e UE | `pSCS`, `pSizeGrid`, `pBW`, `pCyclicPrefix`, `pNCellID`, `pLayers` | Define a numerologia, largura da grade, largura de banda, CP, célula e camadas usadas pelos dois nós. |
| MCS | `DCIModulation` | Índice de MCS; `getMCSLookupTable` converte-o em modulação e taxa de código usadas por gNB e UE. |
| Simulação | `pSimArg.pSNRdB`, `pSimArg.pFrequencyOffset`, `pSimArg.pPhaseOffset` | Configura AWGN, offset de frequência e offset de fase quando o sentido usa `software`. |
| ADALM-PLUTO | `pPlutoArg.pGain`, `pPlutoArg.pCenterFrequency`, `pRadioIDTx`, `pRadioIDRx` | Configura ganho, frequência RF e IPs/IDs do transmissor e receptor do uplink. |
| ADALM2000 | `pM2K.pVLCCenterFrequency`, `pM2K.pDACSampleRate`, `pM2K.pADCSampleRate`, `pM2KTxIP`, `pM2KRxIP` | Configura portadora VLC, taxas dos conversores e dispositivos do downlink. |

Antes de usar hardware, atualize os IPs/IDs e confirme as conexões RF e analógicas.
