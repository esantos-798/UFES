# Emulação do downlink 5G NR em Octave e adaptação para enlace VLC/UWOC

**Relatório técnico, Fases 0 a 2**

| | |
|---|---|
| Projeto | `5g-ofdm-test / src / experiments / half-duplex` |
| Data | 05/10/2026 |
| Ambiente de desenvolvimento e teste | GNU Octave 8.4 (pacotes `signal` e `communications`); resultados reproduzidos na máquina do autor (Octave, Windows) |
| Escopo | `main_downlink.m` e `main_downlink_custom.m`; funções `custom_*`; scripts da Fase 0; módulo UWOC (Fases 1 e 2) |

> Convenção de leitura: **[verificado]** = executado e conferido; **[premissa]** = decisão minha sem acesso ao código original; **[a confirmar]** = depende de arquivo ou informação ainda não disponível.

---

## 1. Resumo executivo

**Objetivo.** (i) Rodar o `main_downlink.m` em Octave, sem o 5G Toolbox, por meio de funções `custom_*`, mantendo o `main_downlink_custom.m` idêntico ao original exceto por `if/else`. (ii) Preparar o canal do enlace para a camada física UWOC: LED branco/azul, água (resposta ao impulso Double Gamma), fotodetector e ruído, tendo o modelo `SimLED_conv.m` como base.

**Resultados principais**

1. A emulação **não reproduz 100%** o downlink original, porque não existem LDPC, PDCCH/PBCH reais nem estimador de canal idêntico ao do toolbox. O BER no caminho custom é **pré-FEC** e não é comparável ao BER do original. A cadeia OFDM, o sincronismo (timing/CFO), a detecção de Cell ID, o estimador de canal e o EVM funcionam e foram verificados (seção 4).
2. Dois bugs no `main_downlink_custom.m` original do autor foram identificados e corrigidos: a faixa do CFO fino estava centrada na estimativa grossa (dupla correção; EVM de 314%) e o demodulador 16/64QAM ignorava a normalização de potência (BER ≈ 0,25 com EVM baixo).
3. `modulate`/`demod` não existem no Octave. Foram criados `custom_modulate` e `custom_demod`, com as fórmulas documentadas do MATLAB. Isso destravou o canal `b2b`. O travamento do `fase0_downlink_metricas.m` era um erro de orientação de vetor (`.'`) que expandia uma matriz de 15360×15360 elementos complexos.
4. Foi criado o módulo UWOC (`uwoc_*`, `led_intensity`): DUC → LED → água → fotodetector → ruído → DDC. A física do LED, da CIR e do fotodetector vem do `SimLED_conv.m`. Foi integrado aos dois mains como `pChannel = 'uwoc-b2b'` (Fase 2).
5. Conclusões físicas do enlace OFDM sobre esse modelo:
   - com os coeficientes do `SimLED_conv.m`, a CIR (rms 1,67 ns) é **um delta a 75 MS/s**; a água praticamente não causa ISI nessa taxa;
   - o OFDM tem PAPR de 14,4 dB, logo o swing rms é só ≈ 95 mA contra 500 mA do NRZ do script original;
   - o EVM em canais limpos tem um piso de ≈ 1% causado pela remoção de DC do script original (subportadora de DC), não pelo receptor.

**Decisões pendentes** (seção 11): lei do fotodetector (`linear` × `square`), confirmação da escala de tempo da CIR no `SimLED_conv.m` e dos coeficientes costeiros do plano original, e obtenção dos arquivos originais (`nrOFDM_CFO.m`, `evm.m`, `detecSSS.m`).

---

## 2. Contexto, escopo e glossário

### 2.1 Cadeia do `main_downlink.m`

```
[A] TX 5G NR     bits -> LDPC -> QAM -> grade (SSB + PDCCH + PDSCH/DMRS) -> OFDM -> waveform (7,68 MS/s, complexo)
[B] Canal        software | adalm-pluto | adalm2000 | b2b | uwoc-b2b (novo)
[C] RX 5G NR     timing (SSB) -> CFO grosso/fino -> OFDM demod -> estimação de canal (DMRS) -> MMSE -> LDPC -> BER/EVM
```

Parâmetros do script: SCS 15 kHz, 25 RB (300 subportadoras, ≈ 4,5 MHz), Cell ID 102, SNR 50 dB, CFO injetado 1,2 kHz, fase 15°, portadora VLC 5 MHz, DAC 75 MS/s.

### 2.2 Restrição de projeto

O `main_downlink_custom.m` deve ser o original acrescido **somente** de `if/else` que desviam para funções `custom_*` quando não há 5G Toolbox. Na Fase 2 o autor autorizou uma exceção: inserir o ramo `'uwoc-b2b'` nos dois mains (seção 8).

### 2.3 Glossário

| Termo | Significado |
|---|---|
| **gNB** | Estação rádio-base 5G (rádio, antena e processamento de banda base). No projeto, o downlink emula gNB → UE. |
| **UE** | Equipamento do usuário. |
| **SSB** | Bloco de sincronismo (PSS + SSS + PBCH); dá timing e Cell ID. |
| **DMRS** | Sinal de referência de demodulação; base da estimação de canal. |
| **CFO** | Desvio de frequência de portadora. |
| **DUC / DDC** | Conversor digital para cima / para baixo (banda base ↔ portadora). |
| **IM/DD** | Modulação de intensidade com detecção direta (LED + fotodetector). |
| **CIR / DGF** | Resposta ao impulso do canal / ajuste Double Gamma. |
| **PAPR** | Razão entre potência de pico e média. |
| **EVM** | Magnitude do vetor de erro (qualidade da constelação). |

---

## 3. Downlink × uplink

Esclarecimento pedido durante o trabalho, com base na leitura de `main_uplink.m`.

**Conceito.** No downlink a gNB difunde o SSB e o UE sincroniza nele. No uplink o UE transmite à gNB, que já conhece a célula; o sincronismo e o canal vêm de SRS e DMRS. O UE tem potência limitada, então o uplink opera tipicamente com SNR menor. O padrão admite DFT-s-OFDM no uplink (menor PAPR); neste script o `nrPUSCHConfig` não ativa essa opção, e pelo default do toolbox o uplink usa CP-OFDM, como o downlink.

| | `main_downlink.m` | `main_uplink.m` |
|---|---|---|
| Grade | SSB + PDCCH + PDSCH | SRS + PUCCH formato 2 + PUSCH |
| Referência de sincronismo | `DLgrid(:,1:4)` (SSB) | `ULGrid(:,1:3)` (só o SRS) |
| CFO grosso | ±20 kHz | ±5 kHz |
| Detecção de PSS/SSS/Cell ID | sim | não existe |
| DMRS | defaults do toolbox | `DMRSAdditionalPosition = 1` (2 símbolos) |
| Codificação | DL-SCH | UL-SCH |
| SNR e canal padrão | 50 dB, `software` | 10 dB (software), `adalm-pluto` |
| Normalização do TX | `/max`, `-mean`, `×2` | só `/max` (`-mean` comentado) |

Consequências: a referência de timing/CFO do uplink é muito mais fraca (só SRS, um símbolo), e, no caminho `software`, a remoção de média do TX e do RX está comentada no uplink, então ali não deve aparecer o piso de EVM da subportadora de DC descrito na seção 9. Um `custom` para o uplink reaproveitaria ≈ 70% do que existe, mas exigiria adivinhar o layout de `mapULGrid`/SRS/PUCCH (funções do laboratório que não foram fornecidas) e não ajuda o objetivo UWOC. Ficou como etapa posterior.

---

## 4. Parte A: reconstrução das funções `custom_*`

### 4.1 Diagnóstico inicial

O caminho custom original era uma emulação, não o mesmo sistema:

| Aspecto | Original (toolbox) | Custom original do autor |
|---|---|---|
| Sinal TX | SSB + PDCCH + PDSCH/DMRS, LDPC, > 14 símbolos | grade 300×14 de QAM aleatório |
| CP | 40 amostras no 1º símbolo de cada meia-subframe, 36 nos demais (512-FFT) | 36 em todos; slot de 7672 amostras (real: 7680) |
| Janela da FFT na demodulação | no meio do CP, com compensação de fase | início do fim do CP |
| Equalização | MMSE por RE com estimativa por DMRS | um escalar `mean(rx./tx)`, usando o `tx` conhecido |
| BER | pós-LDPC | bruto |
| Cell ID | detectado (PSS/SSS) | fixo (`rxCellid = pNCellID`) |

### 4.2 Bugs encontrados no `main_downlink_custom.m` do autor

1. **CFO fino**: a faixa `(coarse-50):1:(coarse+50)` era aplicada sobre `correctedWaveform`, que já havia sido desrotacionada pelo CFO grosso. O resíduo está perto de 0, não perto de 1000 Hz, então o fino escolhia a borda (950 Hz) e piorava o sinal.
   - **[verificado]** QPSK: BER 0,344 e EVM 314%; com a faixa original `-1000:5:1000`: fino = 200 Hz, BER 0, EVM 2,4%.
2. **Normalização do QAM**: o TX normalizava a potência para 1, e o RX usava `qamdemod(..., M)` (constelação não normalizada, níveis ±1, ±3). Em QPSK o limiar é 0 e funciona por acaso; em 16/64QAM os limiares ficavam errados.
   - **[verificado]** 16QAM: BER 0,248 com EVM de 2,44%. O autor havia trocado para QPSK, o que escondia o problema.
3. **Alterações fora de `if/else`** (afetavam também o caminho com toolbox): `pDigitalModulation`/`pModulationLevel` trocados, faixa do CFO fino, `getframe/imwrite`, `a = ...` dos eixos, formato dos prints. Todas revertidas no main regenerado.

### 4.3 Reconstrução

| Original (toolbox / lab) | Custom | Observação |
|---|---|---|
| `nrCarrierConfig` | `custom_nrCarrierConfig` | propriedades usadas; nome inválido gera erro |
| `nrOFDMInfo` | `custom_nrOFDMInfo` | `Nfft = 2^ceil(log2(K/0,85))`, mínimo 128 **[premissa]**: exato para 25 RB (512) |
| `nrOFDMModulate` | `custom_nrOFDMModulate` + `custom_nrSymbolLengths` | CP normal do NR por símbolo; DC no índice K/2 |
| `nrOFDMDemodulate` | `custom_nrOFDMDemodulate` | janela em 0,5·CP com compensação de fase **[premissa]** |
| `nrTimingEstimate` | `custom_nrTimingEstimate` | lógica copiada do código-fonte fornecido (padding, lags ≥ 0, `mag` como vetor); correlação via FFT |
| `nrOFDM_CFO` (lab, não fornecido) | `custom_nrOFDM_CFO` | escrito pelo autor; **[a confirmar]** |
| `initSSB`, `mapDLGrid` (lab) | `custom_downlink_tx` | layout assumido (4.4) |
| `detectPSS`, `detecSSS` (lab) | `custom_detectPSS`, `custom_detecSSS` | PSS/SSS reais da TS 38.211; não geram o gráfico de correlações |
| `nrChannelEstimate` | `custom_nrChannelEstimate` | LS no DMRS, suavização de 3 pilotos, interpolação linear; **não bit-exato** |
| `nrEqualizeMMSE` | MMSE em `custom_downlink_rx` | `conj(H)·rx/(|H|²+nVar)` |
| QAM do `nrPDSCH` | `custom_nrSymbolModulate/Demodulate` | mapeamento Gray da TS 38.211 (QPSK/16/64QAM), sem `qammod/bi2de/biterr` |
| LDPC (DL-SCH) | não existe | BER pré-FEC |
| `evm` (lab, não fornecido) | `custom_evm` | mesma normalização do `comm.EVM` padrão |
| `comm.PhaseFrequencyOffset` | `exp(j(2π·f·t + φ))` | equivalente |

O `main_downlink_custom.m` é gerado programaticamente a partir do original: **0 linhas originais alteradas** (`diff -w`) na versão da Fase 1; na Fase 2, apenas as 2 condições dos gráficos (seção 8).

### 4.4 Premissas de layout do TX **[premissa]**

- Grade de 2 slots (28 símbolos). Símbolo 1: "PDCCH" (QPSK aleatório em 24 RB). Símbolos 2 a 5: SSB com PSS e SSS reais e PBCH aleatório; deslocamento `pbCHoffset = round((K-240)/2)`.
- A posição do SSB foi inferida do código original (`sssRx` na coluna 4; `refGrid = DLgrid(:,1:4)`).
- Slot 2 (colunas 15 a 28): PDSCH; DMRS tipo 1 no símbolo 3 do slot, subportadoras pares, sequência Gold da TS 38.211; esse símbolo não carrega dados (`NumCDMGroupsWithoutData = 2`). O downlink original não define posição adicional de DMRS.
- A posição do PDCCH e a ausência de dados no símbolo de DMRS são hipóteses sobre o toolbox.

### 4.5 Verificação **[verificado]**

| Teste | Resultado |
|---|---|
| Ida e volta OFDM | erro máximo 1,3·10⁻¹⁴; 28 símbolos, 15360 amostras |
| CP e slot | CP [40 36 36 36 36 36 36 40…]; slot = 7680 amostras |
| Tolerância de timing (janela adiantada) | 0% de erro em −5 e −18 amostras; 14% em −30 |
| Detecção de Cell ID sem canal | correta para 0, 1, 2, 102, 335, 500, 1007 |
| Varredura QPSK/16/64QAM × 4 sementes (canal `software`) | timing exato, Cell ID 102 e BER entre 0 e 0,0005 |
| Mapeamento | potência média ≈ 1, ida e volta bit a bit |

### 4.6 Proteção contra erro de configuração

Execuções de QPSK com BER ≈ 0,5 e EVM baixo foram causadas por `pModulationLevel` incompatível com `pDigitalModulation` (por exemplo, QPSK com nível 16: 3903 erros em ≈ 7800 bits, reproduzido). `custom_downlink_rx` agora interrompe com a mensagem *"Ajuste pDigitalModulation e pModulationLevel juntos (QPSK=4, 16QAM=16, 64QAM=64)"*.

### 4.7 Limitações desta parte

BER pré-FEC; estimador de canal não idêntico; `custom_nrOFDM_CFO` e `custom_evm` não comparados com os originais; layout do SSB/PDCCH/DMRS inferido; `nrOFDMDemodulate` e a regra de `Nfft` reconstruídos. O ramo com 5G Toolbox do main não pôde ser executado no sandbox (só análise sintática).

---

## 5. Parte B: cadeia passa-faixa (DUC/DDC) e Fase 0

### 5.1 `custom_modulate` e `custom_demod`

Fórmulas documentadas do MATLAB (Signal Processing Toolbox, método `'qam'`):

```
modulate:  y  = I.*cos(2*pi*fc*t) + Q.*sin(2*pi*fc*t)        (equivale a Re{(I - jQ)·e^{j2πfc t}})
demod:     x1 = filtfilt(b,a, y.*cos(2*pi*fc*t))
           x2 = filtfilt(b,a, y.*sin(2*pi*fc*t)),   [b,a] = butter(5, fc*2/fs)
```

- Preservam a orientação (linha → linha, coluna → coluna), necessária porque o main usa `.'` após o DUC.
- O `demod` implementado segue a fórmula documentada, sem fator 2; a saída é ≈ ½ de I/Q. O main normaliza o RX logo depois, então a escala não afeta o resultado. **[a confirmar]** se o `demod` real aplica ganho extra: `valida_custom_modulate_demod.m` (roda só no MATLAB) imprime a razão de ganho.
- **[verificado]** ida e volta: razão de ganho 2,0000 (esperada), erro residual 0,26%; I=1, Q=0 produz exatamente cos, e I=0, Q=1 produz sin; 99,78% da energia dentro de fc ± 2,5 MHz.
- No main, as quatro chamadas de `modulate`/`demod` (ramos `adalm2000` e `b2b`) foram envolvidas em `if hasModDemod ... else custom_* end`, onde `hasModDemod = exist('modulate') && exist('demod')`.

### 5.2 Defeitos corrigidos nos scripts da Fase 0

| Defeito | Causa | Correção |
|---|---|---|
| `fase0_downlink_metricas` travava no `b2b` | `rxWaveform = rxWaveform.'` transformava a coluna em linha; em `custom_nrOFDM_CFO`, `linha .* coluna` virava uma matriz de 15360×15360 complexos (≈ 3,8 GB); no sandbox o processo foi morto por memória | `rxWaveform(:)` |
| `fase0_baseline_didatico` quebrava ao imprimir a tabela | `numel(modos)` = 6 (matriz 3×2) | `size(modos,1)` |
| Fallback de DUC em `fase0_vlc_passband` | `I·cos − Q·sin` (convenção oposta à do `modulate`); sem passa-baixa e sem fator 2 | substituído por `custom_modulate/demod` |

### 5.3 Baseline da Fase 0 (máquina do autor, semente 42, pré-FEC)

| Mod. | Canal | BER | EVM % | Timing | CFO (Hz) |
|---|---|---|---|---|---|
| QPSK | software | 0 | 1,39 | 10 | 1200 |
| QPSK | b2b | 0 | 1,19 | 0 | 0 |
| 16QAM | software | 0,000385 | 2,91 | 36 | 1200 |
| 16QAM | b2b | 0 | 1,44 | 0 | 0 |
| 64QAM | software | 0,000214 | 1,16 | 68 | 1200 |
| 64QAM | b2b | 0 | 0,41 | 0 | 0 |

Esses valores coincidem com os do sandbox com diferença ≤ 0,01 ponto percentual de EVM. O b2b não tem atraso aleatório nem CFO injetados; por isso timing e CFO são 0.

### 5.4 Avaliação do plano original (`report.pdf`)

Pontos fortes: separação em três blocos, função de canal compartilhada pelos dois mains, escada de validação (b2b → delta → DGF → LED). Ressalvas: (i) `modulate/demod` ausentes no Octave; (ii) o risco "CIR vs duração OFDM" é pequeno para esses coeficientes (seção 7); (iii) o plano não previa a resposta em frequência do LED; (iv) os coeficientes costeiros `C1=0,3958; C2=8,811; C3=1589; C4=51,35` vêm de arquivos não fornecidos (`Simulacao_5G_UWOC.m`, `resposta_impulsiva.m`) e, se `t` estiver em ns, o 2º termo teria pico em ≈ 0,02 ns; **[a confirmar]**.

---

## 6. Parte C: o `SimLED_conv.m` em detalhe

### 6.1 O que é

Script autônomo de enlace óptico IM/DD com **modulação simples** (bits NRZ), não NR. Autoria e data no cabeçalho: Cristiano Tavares, abril/2026. O fluxo é:

```
bits NRZ -> corrente do LED -> curva do LED -> água (CIR) -> fotodetector -> DC block -> ruído -> integrate&dump -> decisão -> BER
```

### 6.2 Modelo do LED

```
I_fin = I_bias + alpha * I_mod                          (corrente, mA)
Lum   = I_fin ./ ( zeta + (I_fin/Lum_max).^(2k) ).^(1/(2k))     (saída óptica, lux)
```

![Curva do LED](figuras/fig1_curva_led.png)

Interpretação **[verificado numericamente]**:

- **Corrente baixa**: `Lum ≈ I / ζ^(1/2k) = 0,4546·I` (trecho linear).
- **Corrente alta**: o termo `(I/Lum_max)^(2k)` domina e `Lum → Lum_max`: `Lum(2500) = 718`, `Lum(10⁶) = 755`.
- ζ define a inclinação inicial e a posição do "joelho"; k, a nitidez da transição.
- O LED precisa de **bias** (aqui 800 mA, que também ilumina): a intensidade não pode ser negativa e o sinal oscila em torno desse ponto.

No ponto de operação do script, `800 + 10·(±50)` = 300 a 1300 mA, `Lum` = 136, 358 e 541; o topo comprime **17,2%** em relação ao trecho linear.

### 6.3 Resposta ao impulso Double Gamma

```
h(t) = C1·t·exp(−C2·t) + C3·t·exp(−C4·t),      t em ns,   0 ≤ t ≤ 16 ns
C = [4,777160  3,335616  0,529837  0,799894],   h = h/sum(h)
```

Modela o alargamento temporal de um pulso de luz na água por espalhamento (ajuste de simulações Monte Carlo, segundo o plano original). Detalhes:

- Amostrada em `Ts = 1/Rs = 1 ns` (17 amostras, de 0 a 16 ns). Média de 2,34 ns e espalhamento rms de **1,67 ns**.
- `sum(h) = 1`: a CIR **não atenua**; só alarga o pulso. A perda do enlace não está no modelo e entra, neste projeto, pela SNR.

![CIR](figuras/fig2_cir_dgf.png)

### 6.4 Fotodetecção, DC block e ruído

```
y   = conv(Lum_fin, h, "same")
Ir  = R * abs(y).^2            % R = 0,6 ; lei quadrática
Irb = Ir - mean(Ir)            % DC block
SNR = SNR_dB + 10*log10(Tempo_pulso)       % 9,6 + 6,0 = 15,6 dB
Ir_ruido = awgn(Irb, SNR, 'measured')
bits = intdump(Ir_ruido, Tempo_pulso) > 0
```

### 6.5 Parâmetros do script

| Parâmetro | Valor | Papel |
|---|---|---|
| `Lum_max` / `zeta` / `k` | 755 / 20 / 1,9 | curva do LED |
| `I_bias` / `I_mod_A` / `alpha` | 800 mA / 50 mA / 10 | polarização e modulação (excursão de pico = α·I_mod_A = 500 mA) |
| `R` | 0,6 | responsividade (cita Camporez et al., 2024) |
| `SNR_dB` | 9,6 | SNR de entrada (o script soma 10·log10(4)) |
| `N_bits`, `Tempo_pulso` | 3000, 4 | bits e amostras por bit |
| `Rb`, `Rs`, `Ts` | 1 Gb/s, = Rb, 1 ns | taxa de bits e amostragem da CIR |
| `T_ns`, `C1..C4` | 16 ns, ver 6.3 | janela e coeficientes da CIR |
| `rng(42)` | n/d | reprodutibilidade |

### 6.6 O que foi aproveitado e onde está

| Bloco do `SimLED_conv.m` | Destino | Mudança |
|---|---|---|
| `Lum_max, zeta, k, I_bias, I_mod_A, alpha` | `uwoc_defaults.m` | nenhuma |
| `I_fin = I_bias + alpha*I_mod`; `Lum_fin` | `led_intensity.m` | acrescentado limite `I ≥ 0` |
| `C1..C4`, janela de 16 ns, `h/sum(h)` | `uwoc_dgf_impulse.m` | modo `'sample'` replica o script **[verificado: erro 0]**; padrão passou a `'integrate'` |
| `R`, `Ir = R*abs(y).^2`, `Irb = Ir - mean(Ir)` | `uwoc_apply.m` | lei selecionável (`'linear'` ou `'square'`) |
| `awgn(..., 'measured')` | `uwoc_apply.m` | ruído gaussiano medido sobre o sinal AC, sem a soma de `10·log10(Tempo_pulso)` |
| `conv(Lum_fin, h, "same")` | `uwoc_apply.m` | filtro causal com início em regime, sem o atraso de meia janela |

### 6.7 O que não foi aproveitado

Geração de bits NRZ, `rectpulse`, `intdump`, contagem de erros, diagrama de olho, varredura da curva `Lum(I)` de 0 a 2500 mA (calculada mas nunca plotada) e as definições de `Rb/Rs/Ts`. No lugar entram o OFDM do NR e o DAC a 75 MS/s.

### 6.8 Diferenças que importam para o enlace OFDM

1. **PAPR.** O OFDM tem PAPR de 14,4 dB. Com o mesmo pico de ±500 mA, o swing rms é ≈ 95 mA (índice de modulação rms 0,119), contra 500 mA do NRZ (índice 0,625). Para o mesmo ruído, a SNR elétrica do OFDM é ≈ 14 dB pior; em compensação, o sinal fica a maior parte do tempo perto do bias, onde a curva do LED é quase linear.
2. **Atenuação.** Não existe no modelo (CIR normalizada).
3. **Lei quadrática** do fotodetector: eleva ao quadrado uma grandeza que o próprio script chama de potência óptica; em IM/DD a fotocorrente costuma ser linear na potência. Medido: com swing de 500 mA, `square` custa ≈ 1,1 ponto percentual de EVM (seção 7.4).

### 6.9 Pontos do `SimLED_conv.m` a confirmar com o autor

- **Escala de tempo da CIR.** O sinal tem 4 amostras por bit a 1 Gb/s (4 GS/s), mas `h` é amostrada a 1 GS/s e a linha que a superamostraria (`h_s = rectpulse(h, Tempo_pulso)`) está comentada. Pelo código, cada amostra de `h` corresponde a 0,25 ns na convolução e a CIR de 16 ns se comporta como de ≈ 4 ns.
- **Lei quadrática** (6.8, item 3).
- **Soma de 10·log10(4)** à SNR (15,6 dB efetivos).
- **Nomes:** o cabeçalho cita `P_max`; o código usa `Lum_max` na razão `(I/Lum_max)`, com `I` em mA. Funcionou como parâmetro de saturação, mas a unidade mistura corrente e lux.

---

## 7. Parte D: Fase 1, módulo UWOC

### 7.1 Arquitetura

![Pipeline](figuras/fig5_pipeline.png)

`uwoc_vlc_chain(waveform, Fs_nr, p)` recebe a banda base NR já normalizada pelo main, executa a cadeia acima e devolve `rxWaveform` (coluna, sem normalizar) mais diagnósticos (`waveform_pb`, corrente, `Lum`, `h`, fração de clipping, índice de modulação, PAPR). Usa `modulate/demod` reais quando existem e `custom_*` caso contrário.

### 7.2 Funções e parâmetros padrão

| Arquivo | Função |
|---|---|
| `uwoc_defaults.m` | parâmetros padrão |
| `led_intensity.m` | modelo estático do LED |
| `uwoc_dgf_impulse.m` | CIR Double Gamma amostrada a Fs (`'sample'` ou `'integrate'`) |
| `uwoc_apply.m` | convolução causal, fotodetecção, DC block e ruído |
| `uwoc_vlc_chain.m` | cadeia completa |
| `fase1_validacao.m` | varreduras A a D com o receptor NR |

| Parâmetro | Padrão |
|---|---|
| `fc`, `Fs_dac` | 5 MHz, 75 MS/s |
| `I_bias`, `I_mod_A`, `alpha` | 800 mA, 50 mA, 10 |
| `Lum_max`, `zeta`, `k` | 755, 20, 1,9 |
| `R`, `pdLaw` | 0,6, `'linear'` |
| `C`, `t_window_ns`, `cirMethod` | coeficientes do `SimLED_conv.m`, 16 ns, `'integrate'` |
| `SNR_dB` | 30 |

### 7.3 Decisões de projeto

- **CIR por integração.** Amostrar a CIR diretamente a 75 MS/s dá `h = [0 1]` (`h(0) = 0` por causa do fator `t`), um atraso espúrio de uma amostra, que custa +0,5 ponto de EVM. Integrar a CIR dentro de cada intervalo de 13,3 ns dá `h = [0,9999 0,00014]`, praticamente um delta: |H| = 1,00000 / 0,99999 / 0,99998 em 2,75 / 5 / 7,25 MHz.
- **Convolução causal** com borda preenchida pelo primeiro valor, para não criar transiente de DC nem atraso de meia janela.
- **Lei do fotodetector `linear` por padrão** (fisicamente usual), com `'square'` disponível para replicar o `SimLED_conv.m`. **[decisão pendente]**
- **Definição de SNR**: potência do sinal AC dividida pela potência do ruído branco em 0 a Fs/2 (37,5 MHz). A SNR efetiva na banda do sinal é ≈ 10 dB maior (medido ≈ 10,4 dB a partir do EVM; razão teórica 37,5 MHz / 3,84 MHz).

### 7.4 Resultados (16QAM, canal `uwoc-b2b`, semente 42, pré-FEC) **[verificado; idêntico no Octave do autor]**

**Referência b2b ideal:** QPSK 1,19%, 16QAM 1,44%, 64QAM 0,41% de EVM.

**A) SNR elétrica (BER / EVM %)**

| Mod. | Inf | 30 dB | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|---|
| QPSK | 0 / 1,6 | 0 / 1,9 | 0 / 3,5 | 0 / 5,7 | 0 / 9,8 | 0 / 17,0 | 3,9·10⁻⁴ / 29,2 |
| 16QAM | 0 / 1,9 | 0 / 2,2 | 0 / 3,6 | 0 / 5,8 | 0 / 9,8 | 3,6·10⁻³ / 17,1 | 5,6·10⁻² / 29,2 |
| 64QAM | 0 / 1,2 | 0 / 1,5 | 0 / 3,3 | 0 / 5,5 | 7,4·10⁻³ / 9,6 | 5,8·10⁻² / 16,8 | 0,147 / 28,8 |

![EVM x SNR](figuras/fig3_evm_snr.png)

**B) Excursão do LED (16QAM, sem ruído)**

| `I_mod_A` (mA) | swing (mA) | clipping % | I_min (mA) | BER | EVM % |
|---|---|---|---|---|---|
| 5 | 50 | 0,00 | 751 | 0 | 1,44 |
| 20 | 200 | 0,00 | 603 | 0 | 1,51 |
| 50 | 500 | 0,00 | 306 | 0 | 1,93 |
| 100 | 1000 | 0,02 | 0 | 0 | 3,45 |
| 150 | 1500 | 0,87 | 0 | 6,4·10⁻⁵ | 7,21 |

Com `I_mod_A = 5` (região linear) o EVM é igual ao do b2b ideal: a cadeia DUC → LED → DDC é transparente quando não há compressão.

**C) Lei do fotodetector (16QAM, sem ruído):** `I_mod_A = 20`: linear 1,51% × square 1,76%; `I_mod_A = 50`: linear 1,93% × square 3,04%.

**D) Amostragem da CIR a 75 MS/s:** `integrate` 1,93% × `sample` 2,43% de EVM.

---

## 8. Parte E: Fase 2, integração nos mains

Alterações em relação ao `main_downlink.m` original (iguais nos dois mains):

1. **Comentário** na linha dos canais e **struct `pUWOC`** após `pDACSampleRate` (`SNR_dB`, `pdLaw`, `I_mod_A`; campos omitidos usam `uwoc_defaults.m`).
2. **Ramo novo** `elseif strcmp(pChannel,'uwoc-b2b')` antes do `else` do b2b: chama `uwoc_vlc_chain` e define `waveform_up`, `waveform_pb`, `rxWaveform_pb` e `rxWaveform_up` na mesma orientação do b2b, para os gráficos de espectro a 75 MS/s.
3. **Duas linhas originais alteradas**: as condições `if strcmp(pChannel,'adalm2000') || strcmp(pChannel,'b2b')` dos gráficos extras, que agora incluem `|| strcmp(pChannel,'uwoc-b2b')`. São as únicas.

**Testes (`main_downlink_custom.m` inteiro, Octave, gráficos substituídos por stubs) [verificado]:**

| Canal | Mod. | Parâmetros | BER | EVM % |
|---|---|---|---|---|
| software | 16QAM | n/d | 0 | 1,09 |
| b2b | 16QAM | n/d | 0 | 1,12 |
| uwoc-b2b | QPSK | SNR 30 dB | 0 | 2,03 |
| uwoc-b2b | 16QAM | SNR 30 dB | 0 | 1,78 |
| uwoc-b2b | 64QAM | SNR 30 dB | 0 | 1,67 |
| uwoc-b2b | 16QAM | SNR 10 dB | 0 | 9,93 |
| uwoc-b2b | 16QAM | sem ruído, `square` | 0 | 2,55 |

A seção de gráficos executa até o fim e as variáveis têm a orientação esperada. O `main_downlink.m` (com toolbox) só passou em análise sintática; **não foi executado**.

---

## 9. Análise das constelações (QPSK, três canais)

Reprodução com a cadeia custom (semente 21), no mesmo layout das figuras do autor.

![Constelações](figuras/fig4_constelacoes.png)

| Canal | Só timing | Timing + freq | + Eq |
|---|---|---|---|
| software | **anel**: o CFO de 1,2 kHz gira a constelação ao longo dos 2 slots | 4 aglomerados girados ≈ 15° (a fase injetada ainda não foi corrigida) | cantos no lugar, pela estimativa de fase do DMRS |
| uwoc-b2b | sem anel (esse ramo não injeta CFO nem fase); alinhado ao TX | aglomerados mais largos: ruído de 30 dB e compressão do LED | idem |
| b2b | 4 pontos compactos | idem | idem |

O RX aparece em ±0,33 contra ±0,7 do TX por causa da normalização do main; a equalização restaura a escala.

**Pontos soltos.** Em todos os canais, os pontos fora dos aglomerados (erro > 0,1) ficam **exclusivamente na subportadora 151, a de DC**: 8 no software, 12 no uwoc-b2b e 12 no b2b, em 3900 símbolos. A causa é o `waveform - mean(waveform)` do script original, que zera o DC e corrompe os símbolos dessa subportadora. Impacto no EVM **[verificado, reprodução]**:

| Canal | EVM total | EVM sem a subportadora de DC |
|---|---|---|
| software | 1,19% | 0,44% |
| b2b | 1,00% | 0,54% |
| uwoc-b2b | 1,86% | 1,65% |

Nos canais limpos o piso de EVM vem quase todo da remoção de DC, não do receptor. Em varreduras de EVM entre canais e SNRs, isso mascara diferenças pequenas.

---

## 10. Limitações e itens em aberto

1. **Arquivos originais ausentes**: `nrOFDM_CFO.m`, `evm.m` (o enviado é a classe `comm.EVM`, não a função `evm(M,tx,rx)` do main) e `detecSSS.m` (gera o gráfico de correlações SSS, que o custom não reproduz).
2. **BER pré-FEC** em todo o caminho custom; não comparável a resultados com LDPC.
3. **Premissas** de layout do TX, `Nfft`, janela da FFT e estimador de canal (4.3 e 4.4).
4. **Ramo com 5G Toolbox** dos mains não executado; só análise sintática.
5. **`custom_demod`**: ganho do `demod` real não confirmado (script de validação no MATLAB pendente).
6. **Modelo UWOC**: sem resposta em frequência do LED, sem atenuação da água (somente via SNR), CIR curta demais para aparecer a 75 MS/s, e a escala de tempo da CIR no `SimLED_conv.m` precisa ser confirmada (6.9).
7. **Coeficientes costeiros** do plano original não usados (5.4).
8. **Uplink**: sem `custom` para `main_uplink.m`.

---

## 11. Decisões pendentes e próximos passos

| Prioridade | Item | Observação |
|---|---|---|
| 1 | Definir a lei do fotodetector padrão (`linear` × `square`) | `pdLaw` já é parâmetro; muda o EVM em ≈ 1,1 ponto no swing padrão |
| 2 | Confirmar com o autor do `SimLED_conv.m` a escala de tempo da CIR e a lei quadrática | 6.9 |
| 3 | Enviar `Simulacao_5G_UWOC.m` e `resposta_impulsiva.m` | conferir unidade e valores dos coeficientes costeiros |
| 4 | Coluna "EVM sem DC" na tabela da Fase 0 e no `fase1_validacao.m` | sem alterar os mains |
| 5 | Fase 3: varreduras de modulação e SNR no próprio main, comparáveis ao trabalho terrestre | |
| 6 | Rodar `valida_custom_modulate_demod.m` no MATLAB e testar o main com toolbox | fecha as pendências 4 e 5 da seção 10 |
| 7 | Modelar a resposta em frequência do LED (passa-baixa) | provável fonte de seletividade em frequência no enlace real |
| 8 | `custom` do uplink | só se DL e UL completos em Octave forem necessários |

---

## Apêndice A: inventário de arquivos

**`octave_5g/`**: `main_downlink_custom.m` (original + `if/else` + ramo UWOC), `custom_nrCarrierConfig`, `custom_nrOFDMInfo`, `custom_nrSymbolLengths`, `custom_nrOFDMModulate`, `custom_nrOFDMDemodulate`, `custom_nrTimingEstimate`, `custom_nrOFDM_CFO` (inalterado), `custom_nrSymbolModulate`, `custom_nrSymbolDemodulate`, `custom_nrPSS`, `custom_nrSSS`, `custom_detectPSS`, `custom_detecSSS`, `custom_nrChannelEstimate`, `custom_downlink_tx`, `custom_downlink_rx`, `custom_evm` (inalterado), `custom_modulate`, `custom_demod`, `valida_custom_modulate_demod.m`.

**`fase0/`**: `fase0_baseline_didatico.m` (correção de `numel`), `fase0_downlink_metricas.m` (correção de orientação, DUC/DDC custom, canal `uwoc-b2b`), `fase0_vlc_passband.m` (inalterado).

**`uwoc/`**: `uwoc_defaults`, `led_intensity`, `uwoc_dgf_impulse`, `uwoc_apply`, `uwoc_vlc_chain`, `fase1_validacao`.

**`fase2/`**: `main_downlink.m` (original + as mesmas inserções UWOC).

Colocar todos na mesma pasta (`half-duplex`) ou no path do Octave.

## Apêndice B: como executar

```
octave-cli .\fase0_baseline_didatico.m      % tabela da Fase 0
octave-cli .\fase1_validacao.m              % varreduras A a D do canal UWOC
octave-cli .\main_downlink_custom.m         % pChannel = 'uwoc-b2b' no topo do script
```

Em `main_downlink_custom.m`, escolher `pChannel = 'uwoc-b2b'` e ajustar `pUWOC`. Ao trocar a modulação, alterar `pDigitalModulation` e `pModulationLevel` juntos.

## Apêndice C: reprodutibilidade

- Scripts da Fase 0 e 1: `rng(42)`; resultados idênticos entre o sandbox e a máquina do autor.
- Sandbox: Octave 8.4 com `signal` e `communications`; gráficos substituídos por stubs nos testes automáticos.
- Figuras 1, 2, 3 e 5 geradas a partir das fórmulas e das tabelas deste relatório; figura 4 gerada a partir de dados exportados da simulação (semente 21).
