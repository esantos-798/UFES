# Utilitários de experimento

Scripts auxiliares para inspeção de sinais e estruturas 5G NR.

- `ssburst_teste.m` e `ssbteste.m`: testes de SSB.
- `plotSSB.m`: visualização do SSB.
- `exemploCR.m`: exemplo relacionado à taxa de código.

Execute cada script individualmente no MATLAB. Eles são apoio aos experimentos principais e podem exigir as toolboxes listadas no README da raiz.

## Parametrização

Estes scripts não configuram uma bancada SDR. Os parâmetros relevantes são os de numerologia e de visualização definidos dentro de cada arquivo:

| Script | Parâmetros a revisar | Descrição |
| --- | --- | --- |
| `ssburst_teste.m` / `ssbteste.m` | configuração de portadora e `NCellID` | Determina a numerologia e a identidade de célula usadas para gerar o SSB. |
| `plotSSB.m` | grade ou sinal de entrada | Define a estrutura 5G NR exibida no gráfico. |
| `exemploCR.m` | taxa de código e modulação | Controla o exemplo de codificação e sua representação. |
