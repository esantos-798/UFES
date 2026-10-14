# 5G NR OFDM Test

Implementação em MATLAB de formas de onda OFDM 5G NR e de experimentos em simulação, RF e comunicação por luz visível (VLC). Os experimentos usam ADALM-PLUTO para o enlace RF e ADALM2000 para o enlace VLC.

## Requisitos

- MATLAB com 5G Toolbox, Communications Toolbox e Signal Processing Toolbox.
- ADALM-PLUTO Support Package para os experimentos RF.
- libm2k configurada no MATLAB e dois ADALM2000 para a bancada VLC.

Para iniciar pelo modo sem hardware, abra o MATLAB em `src/`, entre no experimento desejado e selecione `software` como canal.

## Elementos de canal

| Valor | Elemento | Aplicação |
| --- | --- | --- |
| `software` | MATLAB | AWGN, atraso e desvios de fase/frequência; recomendado para validação inicial. |
| `adalm-pluto` | ADALM-PLUTO | Enlace RF por `sdrtx` e `sdrrx`. |
| `adalm2000` | ADALM2000 | Enlace VLC em banda passante via DAC/ADC e libm2k. |
| `b2b` | Loopback em software | Valida a conversão para banda passante, sem hardware. |

No enlace integrado, o downlink aceita `software` ou `adalm2000`; o uplink aceita `software` ou `adalm-pluto`.

## Experimentos

| Experimento | Objetivo | Documentação |
| --- | --- | --- |
| Half-duplex | Geração, transmissão, sincronização e decodificação isoladas de downlink e uplink. | [src/experiments/half-duplex](src/experiments/half-duplex/README.md) |
| Full-duplex | Downlink gNB→UE e ACK/uplink UE→gNB no mesmo fluxo. | [src/experiments/full-duplex](src/experiments/full-duplex/README.md) |
| Resposta em frequência | Varreduras com ADALM-PLUTO e ADALM2000. | [src/experiments/freqResponse](src/experiments/freqResponse/README.md) |
| VLC Test Bed | Varredura automatizada de MCS, corrente de polarização e distância. | [src/experiments/VLCTestBed](src/experiments/VLCTestBed/README.md) |
| Utilitários de experimento | Testes e visualizações auxiliares de SSB e taxa de código. | [src/experiments/etc](src/experiments/etc/README.md) |

## Parametrização comum

| Grupo | Parâmetros | Função |
| --- | --- | --- |
| NR | `pSCS`, `pSizeGrid`, `pLayers`, `pCyclicPrefix`, `pNCellID` | Define espaçamento de subportadoras, RBs, camadas, CP e célula. |
| Modulação/código | `pDigitalModulation`, `pModulationLevel`, `pCodeRate`, `DCIModulation` | Define QPSK/16QAM/64QAM e a taxa de código/MCS. |
| Simulação | `pSNRdB`, `pFrequencyOffset`, `pPhaseOffset` | Controla AWGN e imperfeições do canal simulado. |
| ADALM-PLUTO | `pGain`, `pCenterFrequency`, `pRadioIDTx`, `pRadioIDRx` | Configura ganho, frequência e dispositivos RF. |
| ADALM2000 | `pVLCCenterFrequency`, `pDACSampleRate`, `pADCSampleRate` | Configura portadora VLC e conversores DAC/ADC. |

Cada experimento documenta os seus parâmetros específicos ao lado do script que os usa.

## Conteúdo versionado

O repositório mantém scripts MATLAB e documentação. Resultados, formas de onda, arquivos `.mat`, `.fig`, imagens, PDFs e autosaves permanecem locais e são ignorados pelo Git.
