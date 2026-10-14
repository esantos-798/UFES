# Resposta em frequência

Mede e visualiza a resposta em frequência das bancadas RF e VLC.

- `mainPluto.m`: varredura com ADALM-PLUTO.
- `mainPLuto2Antenas.m`: varredura PLUTO com duas antenas.
- `mainM2k.m`: varredura ADALM2000/VLC.
- `plutoFreqTest*.m`: variantes de teste e correção para PLUTO.
- `ofdm3D.m`: visualização OFDM.

## Parametrização

| Script | Parâmetros | Descrição |
| --- | --- | --- |
| `mainPluto.m` / `mainPLuto2Antenas.m` | `fmin`, `fmax`, `fpass` | Frequência inicial, final e passo da varredura RF. |
| `mainPluto.m` / `mainPLuto2Antenas.m` | `pGain`, `pRadioIDTx`, `pRadioIDRx` | Ganho de transmissão e identificação USB/IP dos ADALM-PLUTO. |
| `mainM2k.m` | `pM2K.pDACSampleRate`, `pM2K.pADCSampleRate` | Taxas de amostragem do gerador e do osciloscópio ADALM2000. |
| `mainM2k.m` | `fmin`, `fmax`, `fpass` | Define a faixa e a resolução da varredura VLC. Passos menores aumentam o tempo de aquisição. |
| `mainM2k.m` | `ain.setRange(...)` | Define a faixa de tensão de entrada do ADC; ela deve acomodar o sinal recebido sem saturação. |

## Como rodar

Abra o script correspondente à bancada. Ajuste frequência inicial/final, `fpass` (passo), taxa de amostragem e identificação dos dispositivos antes de executar.

Para o PLUTO, confira frequência central, ganho e IDs/IPs. Para o ADALM2000, confira `pM2K.pDACSampleRate`, `pM2K.pADCSampleRate`, faixas analógicas e a calibração DAC/ADC.
