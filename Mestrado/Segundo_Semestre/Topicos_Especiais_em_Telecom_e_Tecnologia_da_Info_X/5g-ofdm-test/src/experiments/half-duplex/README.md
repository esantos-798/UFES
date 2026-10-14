# Half-duplex

Executa os sentidos do enlace 5G NR separadamente.

- `main_downlink.m`: SSB, PDCCH e PDSCH; sincronização e decodificação de downlink.
- `main_uplink.m`: SRS, PUCCH e PUSCH; sincronização e decodificação de uplink.
- `main_downlink_branch1.m`: variante de downlink.

## Como rodar

No MATLAB, entre nesta pasta, abra um dos scripts principais e execute-o. Para validar sem equipamentos, defina `pChannel = 'software'`.

## Parametrização

Todos os parâmetros estão no início de `main_downlink.m` e `main_uplink.m`.

| Grupo | Parâmetros | Descrição |
| --- | --- | --- |
| Canal | `pChannel` | Escolhe `software`, `adalm-pluto`, `adalm2000` ou `b2b`. Use `software` antes de conectar hardware. |
| Grade NR | `pSCS`, `pSizeGrid`, `pLayers`, `pCyclicPrefix`, `pNCellID` | Define espaçamento entre subportadoras (kHz), número de RBs, número de camadas, tipo de CP e identificador da célula. |
| Payload | `pDigitalModulation`, `pModulationLevel`, `pCodeRate` | Define modulação (QPSK/PSK ou QAM), ordem da modulação e taxa de codificação. |
| Simulação | `pSNRdB`, `pFrequencyOffset`, `pPhaseOffset` | Define nível de AWGN, desvio de frequência em Hz e desvio de fase em graus. Só afeta `software`. |
| ADALM-PLUTO | `pGain`, `pCenterFrequency`, `pRadioIDTx`, `pRadioIDRx` | Configura ganho do transmissor, frequência RF e IDs USB/IP dos rádios. |
| ADALM2000 | `pVLCCenterFrequency`, `pDACSampleRate`, `pADCSampleRate` | Configura portadora elétrica VLC e taxas de conversão do DAC e ADC. |

Use `adalm-pluto` para RF, `adalm2000` para VLC e `b2b` para a validação de banda passante sem hardware.
