# PAPEL
Você é um designer de apresentações técnicas e professor de redes de computadores. Crie uma apresentação HTML interativa e animada sobre o artigo anexo (DCell, SIGCOMM'08), em português do Brasil.

# CONTEXTO
- Público: Colegas de mestrado em Eng. Elétrica / telecom  e professor especialista em Datacenter e Redes de Computadores
- Nível de conhecimento prévio do público: Diversificado - alguns conhecem redes e roteamento, mas há quem nunca viu topologias de data center. O único especialista é o próprio professor responsável pela disciplina.
- Ocasião: Seminário de disciplina referente a Datacenter
- Duração: 50 min de fala + 10 min de perguntas. NUNCA ultrapassar 60 min no total.
- Anexos: PDF do artigo (única fonte de fatos) e dcell.py (implementação de referência, já validada contra o artigo).

# FILOSOFIA DOS SLIDES
- Mostre, escreva apenas o essencial. Cada slide carrega UMA ideia.
- Máximo 12 palavras de texto por slide (títulos incluídos). Sem parágrafos e sem listas com mais de 3 itens.
- Prefira diagramas, números grandes, animações e demonstrações a texto.
- Todo o texto que eu falaria vai nas NOTAS DO APRESENTADOR, não no slide.
- Animações devem explicar (construir a estrutura aos poucos, destacar o caminho de um pacote), nunca decorar. Cada clique/tecla revela um passo.

# ESTRUTURA E TEMPO (total ~50 min, 26 a 30 slides)
1. Abertura e problema (4 min): data centers crescendo (450 mil servidores do Google em 2006; Microsoft dobrando a cada 14 meses) e os 3 objetivos: escala, tolerância a falhas, alta capacidade.
2. Por que a árvore de switches não basta (4 min): animar o gargalo no topo e o ponto único de falha (um switch central cai e milhares de servidores ficam isolados).
3. A ideia do DCell e sua construção (9 min): DCell0 (n servidores + mini-switch), DCell1 (n+1 DCell0 ligados todos-com-todos pela regra [i,j-1]↔[j,i]), recursão para níveis maiores. Mostrar o grau do servidor (k+1).
4. Escala e propriedades (4 min): crescimento duplamente exponencial (n=4: 4, 20, 420, 176.820; n=6, k=3: 3,26 milhões), diâmetro ≤ 2^(k+1)−1, bisseção.
5. Roteamento sem falhas, o DCellRouting (6 min): divisão e conquista, com o exemplo n=2, k=2, de [0,2,1] a [1,2,1] (7 saltos, enquanto o mínimo é 3).
6. Tolerância a falhas, o DFR (8 min): local-reroute com proxy, link-state local, jump-up para falha de rack, retry count e TTL.
7. Expansão incremental (3 min): abordagem de baixo para cima (frágil) versus de cima para baixo (DCell1 como unidade).
8. Resultados (6 min): simulações (DFR vs menor caminho) e testbed de 20 servidores (DCell ~2x mais rápido que a árvore: 2270 s vs 4460 s; 9,6 vs 3,6 Gb/s).
9. Comparação e limitações (3 min): Tabela 3 resumida em visual, e a limitação do cabeamento mais longo e numeroso.
10. Conclusão (3 min): 3 mensagens para levar para casa, mais perguntas.

# DEMONSTRAÇÕES INTERATIVAS (5, todas embutidas nos slides)
A. Construtor de DCell: sliders para n (2–6) e k (0–3); animação de blocos se multiplicando e contador de servidores t_k em tempo real.
B. Roteamento passo a passo: o público escolhe origem e destino em um DCell pequeno (n=2, k=2); a tecla → revela cada salto e a recursão (src→n1, cabo, n2→dst).
C. Quebre um cabo: clicar em qualquer link do caminho o "quebra"; a animação mostra o proxy escolhido, o desvio e o contador de saltos (7 → 9 no exemplo). Usar a lógica do dcell.py (route_with_failed_link), traduzida para JavaScript.
D. Corrida de vazão: barras animadas DCell vs árvore (9,6 vs 3,6 Gb/s) e os tempos de conclusão (2270 s vs 4460 s).
E. Mini-quiz de 1 pergunta antes da conclusão: "qual desses links causa mais tráfego?" (nível 0; Teorema 5).

# REQUISITOS TÉCNICOS
- Um único arquivo .html autocontido (CSS e JS embutidos). Se usar bibliotecas, apenas por CDN com versão fixa. Sem imagens remotas.
- Navegação por setas do teclado e clique; barra de progresso; botão de tela cheia.
- Tecla N alterna as notas do apresentador com roteiro e cronometragem por slide (tempo previsto e acumulado).
- Contador de tempo no canto que avisa aos 45 min.
- Funcionar em projetor (alto contraste, fonte mínima 28 px no corpo, 48 px em números de destaque).
- Tema escuro, uma cor de destaque para o "caminho do pacote" e outra para "falha". Paleta consistente em todos os slides.
- Todas as animações devem ter respeito a prefers-reduced-motion.

# REGRAS DE FIDELIDADE
- Todo número, teorema e resultado deve vir do artigo. Não invente dados. Se algo não estiver no artigo, não apresente como fato.
- Diferencie claramente o que é resultado do artigo do que é interpretação minha. Se citar o dcell.py, diga que é uma implementação independente para fins didáticos.
- Não reproduza trechos longos do artigo nem figuras copiadas; redesenhe os diagramas.
- Cite o artigo no slide inicial e no final: Guo et al., "DCell: A Scalable and Fault-Tolerant Network Structure for Data Centers", SIGCOMM 2008.

# ENTREGA EM DUAS ETAPAS
Etapa 1 (agora): entregue apenas o ROTEIRO: tabela com nº do slide, título, ideia única, elemento visual/animação, interação (se houver), tempo previsto e tempo acumulado. Confirme que a soma não passa de 50 min. Aguarde minha aprovação.
Etapa 2 (após eu aprovar): gere o arquivo HTML completo, com notas do apresentador em cada slide.

Ao final da Etapa 2, liste: (a) o que testar antes de apresentar, (b) atalhos de teclado, (c) o que cortar se o tempo apertar (com os slides em ordem de prioridade de corte).