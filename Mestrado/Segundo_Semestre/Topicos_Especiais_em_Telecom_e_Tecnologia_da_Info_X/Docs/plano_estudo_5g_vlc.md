# Plano de Estudo: 5G, Numerologia e OWC/VLC

Este documento fornece um roteiro estruturado para compreender os fundamentos do 5G, sua numerologia, e como essas tecnologias se integram a sistemas de Comunicação Óptica Sem Fio (OWC/VLC), preparando a base teórica para a criação da sua simulação.

## Módulo 1: Fundamentos do 5G NR (New Radio) e OFDM
- **O que é OFDM (Orthogonal Frequency-Division Multiplexing):** Entenda como a informação é dividida em múltiplas subportadoras ortogonais, evitando interferência intersimbólica (ISI).
- **CP-OFDM (Cyclic Prefix OFDM):** O papel do Prefixo Cíclico para mitigar o atraso de propagação (multipath) nos canais.
- **Resource Grid do 5G:** Entenda o conceito de Resource Elements (REs) e Resource Blocks (RBs - blocos de 12 subportadoras).

## Módulo 2: Numerologia do 5G NR
No 4G (LTE), o espaçamento de subportadoras era fixo em 15 kHz. O 5G introduziu a **Numerologia flexível**, permitindo adaptar o sinal para diferentes frequências (FR1 e FR2/Ondas Milimétricas) e cenários de latência.
- **Parâmetro $\mu$ (Numerologia):** O espaçamento de subportadoras é dado por $\Delta f = 2^\mu \times 15 \text{ kHz}$.
    - $\mu = 0 \rightarrow 15 \text{ kHz}$ (Padrão semelhante ao LTE, longo alcance)
    - $\mu = 1 \rightarrow 30 \text{ kHz}$
    - $\mu = 2 \rightarrow 60 \text{ kHz}$
    - $\mu = 3 \rightarrow 120 \text{ kHz}$ (Usado para baixa latência e altas frequências)
- **Relação Tempo/Frequência:** Quanto maior o espaçamento (maior $\mu$), **menor** será a duração do símbolo OFDM no tempo. Isso é crucial para aplicações de Ultra-Low Latency (URLLC).
- **Impacto no Prefixo Cíclico:** O tempo do CP também encurta proporcionalmente.

## Módulo 3: Optical Wireless Communication (OWC) e VLC
Como você irá simular OFDM sobre luz visível, precisa entender as peculiaridades do canal óptico em comparação ao RF.
- **IM/DD (Intensity Modulation and Direct Detection):** Em LEDs, não modulamos a fase ou frequência eletromagnética, apenas a intensidade (potência óptica). Portanto, o sinal OFDM precisa ser **Real e Positivo** (ex: DCO-OFDM ou ACO-OFDM).
- **Modelagem do LED:** O LED possui uma região de operação não-linear que causa distorção (clipping). No seu projeto, isso é modelado pela curva geométrica com os parâmetros $\zeta$ e $k$ (luminância máxima e saturação).
- **Canal Óptico / Resposta ao Impulso:** Em ambientes indoor, a luz sofre reflexões nas paredes, modelado frequentemente por uma resposta *Double Gamma* (que dispersa o sinal no tempo).

## Módulo 4: Hands-on / Implementação em MATLAB
- **5G Toolbox:** Estudar funções como `nrOFDMModulate` e `nrOFDMDemodulate`.
- **Adaptação para VLC:** Estudar como adicionar um *DC Bias* ao sinal OFDM para torná-lo positivo antes de passá-lo pela curva de não-linearidade do LED.

---

# Análise do Projeto Atual

Após analisar os arquivos no seu diretório `c:\UFES\...`, aqui está um resumo do que você já possui e do que trata o projeto:

## 1. Repositório `5g-ofdm-test`
Trata-se de um arcabouço robusto e versionado (provavelmente derivado da dissertação de Augusto Cesar Federici Peterle) feito em MATLAB, desenhado para gerar, transmitir e decodificar sinais 5G NR OFDM.
- **Modularidade:** Ele suporta simulações puramente em software (com AWGN, desvios de fase) e testes com hardware real (ADALM-PLUTO para RF e ADALM2000 para VLC).
- **Experimentos contidos:** Tem fluxos prontos para half-duplex, full-duplex, e notavelmente um **VLC Test Bed** (`src/experiments/VLCTestBed`), que automatiza varreduras de MCS e corrente de polarização de LEDs.

## 2. Scripts Autônomos de Canal Óptico (Ex: `SimLED_conv.m`)
Estes scripts parecem ser o coração da modelagem física do canal OWC. O script `SimLED_conv.m` faz:
- **Modelo de LED Não-Linear:** Aplica uma fórmula para obter a Luminância com base numa corrente, saturando quando chega perto do limite ($Lum_{max}$).
- **Modulação:** Atualmente aplica modulação simples NRZ Bipolar num trem de pulsos retangulares.
- **Resposta ao Impulso Double Gamma:** Gera uma resposta de dispersão temporal típica de canais ópticos difusos, convoluindo-a com o sinal do LED.
- **Recepção:** Modela um fotodetector simples com responsividade $R$, remove o nível DC, aplica Ruído Branco (AWGN) baseado numa SNR desejada, e extrai o BER.

## 3. Dissertação e Notações
- **Dissertação:** O arquivo `dissertacao_de_mestrado_-_augusto_cesar_federici_peterle_final.pdf` serve como a base teórica e de documentação dos experimentos de RF e VLC contidos no repositório `5g-ofdm-test`.
- **Notações:** Imagens (`Optical_Wirelles_Comm`, `Figura_1/2`) provavelmente contêm diagramas de blocos de sistemas IM/DD e esquemáticos das respostas ao impulso e configuração do LED.

---

# Próximo Passo: Integração (A sua simulação)

Para criar a simulação que você deseja, o caminho será **unir o processamento de sinal do `5g-ofdm-test` com a modelagem de canal do `SimLED_conv.m`**.

**Passos lógicos para o seu código futuro:**
1. Usar a base do 5G (Numerologia, Resource Grid) para gerar um vetor de sinal OFDM no domínio do tempo (Complexo).
2. Converter esse sinal para real e positivo (ex: aplicando simetria Hermitiana antes da IFFT para deixá-lo Real, e somar um $I_{bias}$ para deixá-lo Positivo).
3. Passar esse sinal real/positivo pelo modelo matemático de degradação do LED (presente em `SimLED_conv.m`).
4. Aplicar a convolução com a resposta Double Gamma.
5. Adicionar AWGN.
6. No receptor: remover o bias DC, demodular o OFDM e analisar a Constelação e o EVM (Error Vector Magnitude).
