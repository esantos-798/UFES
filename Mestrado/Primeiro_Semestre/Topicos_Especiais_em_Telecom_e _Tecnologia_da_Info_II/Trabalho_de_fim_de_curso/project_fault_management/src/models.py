# src/models.py
import torch
import torch.nn as nn
#import fns
import torch.nn.functional as F

class VanillaRNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class LSTNetModified(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim=64, cnn_kernel=3, skip_period=4):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.skip_period = skip_period
        
        # 1. Camada Convolucional (Padrões de curto prazo)
        # Transforma (Batch, Window, Features) -> (Batch, 1, Window, Features) para Conv2d
        self.conv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, 
                              kernel_size=(cnn_kernel, input_dim))
        self.conv_relu = nn.ReLU()
        
        # Com o kernel_size horizontal igual a input_dim, a saída da conv terá dimensão ideal
        self.output_length = window_size - cnn_kernel + 1
        
        # 2. Camada GRU Tradicional (com ativação ReLU no estado oculto conforme artigo)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.gru_relu = nn.ReLU() # Para aplicar no estado oculto de saída
        
        # 3. Camada Recurrent-Skip (Dependências sazonais de longo prazo)
        self.skip_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Camada densa para combinar as saídas profundas (GRU + Skip)
        # O número de passos que entram na skip_gru depende do skip_period
        self.linear_deep = nn.Linear(hidden_dim + hidden_dim, input_dim)
        
        # 4. Modelo Autoregressivo Linear (Componente Linear AR)
        self.ar_linear = nn.Linear(window_size, 1)
        
        # 5. Modificação Especial do Artigo: Camada Totalmente Conectada com ReLU na saída
        self.final_fc = nn.Linear(input_dim, input_dim)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        
        # --- COMPONENTE CONVOLUCIONAL ---
        x_conv = x.unsqueeze(1) # (Batch, 1, Window, Features)
        c = self.conv(x_conv)
        c = self.conv_relu(c)
        c = c.squeeze(3) # (Batch, Hidden_Dim, Output_Length)
        c = c.permute(0, 2, 1) # (Batch, Output_Length, Hidden_Dim)
        
        # --- COMPONENTE RECORRENTE (GRU) ---
        _, h_gru = self.gru(c)
        h_gru = self.gru_relu(h_gru.squeeze(0)) # (Batch, Hidden_Dim)
        
        # --- COMPONENTE RECORRENTE-SKIP ---
        # Organiza os dados pulando passos temporais fixos (skip_period)
        skip_len = self.output_length // self.skip_period
        if skip_len > 0:
            s = c[:, -skip_len * self.skip_period:, :]
            s = s.view(batch_size, skip_len, self.skip_period, self.hidden_dim)
            s = s.permute(0, 2, 1, 3).contiguous()
            s = s.view(batch_size * self.skip_period, skip_len, self.hidden_dim)
            
            _, h_skip = self.skip_gru(s)
            h_skip = h_skip.squeeze(0)
            h_skip = h_skip.view(batch_size, self.skip_period * self.hidden_dim)
            # Reduz linearmente para casar com a dimensão da GRU padrão
            h_skip_proj = nn.Linear(self.skip_period * self.hidden_dim, self.hidden_dim).to(x.device)(h_skip)
        else:
            h_skip_proj = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            
        # Combina os caminhos profundos
        out_deep = torch.cat((h_gru, h_skip_proj), dim=1)
        res_deep = self.linear_deep(out_deep) # (Batch, Input_Dim)
        
        # --- COMPONENTE AUTOREGRESSIVA (AR) ---
        # Aplica uma regressão linear diretamente na janela histórica de cada variável
        x_ar = x.permute(0, 2, 1) # (Batch, Features, Window)
        res_ar = self.ar_linear(x_ar) # (Batch, Features, 1)
        res_ar = res_ar.squeeze(2) # (Batch, Features)
        
        # --- INTEGRAÇÃO PROPOSTA NO ARTIGO ---
        # Somatório das previsões (Deep + Linear)
        combined_output = res_deep + res_ar
        
        # Acoplamento não-linear via camada Densa + ReLU final
        final_pred = self.final_fc(combined_output)
        final_pred = self.final_relu(final_pred)
        
        return final_pred   

class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=15):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (Batch, Window, Features)
        out, _ = self.gru(x)
        # Pega apenas o último passo temporal da janela para prever o próximo ponto
        out = self.fc(out[:, -1, :])
        return out

class BiLSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=15):
        super().__init__()
        # bidirectional=True dobra a dimensão de saída da camada recorrente
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Hidden_Dim)
        # Calcula a pontuação de importância para cada passo temporal
        attn_scores = self.attn_weights(x) # (Batch, Seq_Len, 1)
        attn_predictions = F.softmax(attn_scores, dim=1) # Normaliza no tempo
        
        # Multiplica os pesos pelos estados originais (Context Vector)
        context = x * attn_predictions # (Batch, Seq_Len, Hidden_Dim)
        return context, attn_predictions

class AttentionLSTNet(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim=64, conv_out_channels=32, kernel_size=3):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        
        # 1. Componente Convolucional Espaço-Temporal
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=conv_out_channels, kernel_size=kernel_size, padding=1)
        
        # 2. Mecanismo de Atenção Temporal
        self.attention = TemporalAttention(hidden_dim=conv_out_channels)
        
        # 3. Componente Recorrente Profundo (GRU)
        self.gru = nn.GRU(input_size=conv_out_channels, hidden_size=hidden_dim, batch_first=True)
        
        # Camada densa de saída para a parte profunda
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # 4. Componente Autorregressivo Linear (Caminho Clássico da LSTNet)
        self.ar = nn.Linear(window_size, 1)

    def forward(self, x):
        # x shape original: (Batch, Window, Features)
        batch_size = x.size(0)
        
        # --- Caminho Profundo ---
        # Conv1d espera (Batch, Channels, Length) -> Permutamos as características para canais
        x_conv = x.permute(0, 2, 1)
        c = F.relu(self.conv(x_conv)) # (Batch, conv_out_channels, Window)
        c = c.permute(0, 2, 1) # Retorna para (Batch, Window, conv_out_channels)
        
        # Aplica Atenção Temporal nos mapas de características da Convolução
        attn_context, weights = self.attention(c)
        
        # Passa pela camada recorrente
        r, _ = self.gru(attn_context)
        r = self.fc(r[:, -1, :]) # Pega o último estado oculto (Batch, Features)
        
        # --- Caminho Autorregressivo (Linear) ---
        # Isola cada feature ao longo do tempo e passa pelo modelo linear
        x_ar = x.permute(0, 2, 1) # (Batch, Features, Window)
        out_ar = self.ar(x_ar) # (Batch, Features, 1)
        out_ar = out_ar.squeeze(-1) # (Batch, Features)
        
        # Fusão das duas componentes (Inovação + Estabilidade)
        output = r + out_ar
        return output

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=15):
        super().__init__()
        # Projeta as features de entrada para a dimensão interna do Transformer (hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Encoder do Transformer do PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Camada linear final para predição
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (Batch, Window, Features)
        x_proj = self.input_projection(x) # (Batch, Window, Hidden_Dim)
        
        # Passa pelo bloco do Transformer
        attn_out = self.transformer_encoder(x_proj) # (Batch, Window, Hidden_Dim)
        
        # Pega o último passo temporal para prever o próximo ponto físico da rede
        out = self.fc(attn_out[:, -1, :]) # (Batch, Output_Dim)
        return out
    
class GraphTemporalFusion(nn.Module):
    def __init__(self, num_nodes, in_features=1, hidden_dim=64, output_dim=15):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Criação da Matriz de Adjacência Fixa (Topologia Linear da Cascata)
        # Se os dados possuem 15 features (ex: 5 amplificadores com 3 métricas cada),
        # podemos mapear a interdependência entre os 15 canais de monitoramento.
        adj = torch.eye(num_nodes)
        for i in range(num_nodes - 1):
            adj[i, i+1] = 1.0
            adj[i+1, i] = 1.0 # Grafo não-direcionado para espalhar correlação
            
        # Normalização do Grafo: D^(-1/2) * A * D^(-1/2)
        degree = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_mat = torch.diag(deg_inv_sqrt)
        
        # Matriz de transição de grafos normalizada e fixa
        self.register_buffer('A_norm', D_mat @ adj @ D_mat)
        
        # Camada de Grafo (Projeção Linear dos nós)
        self.graph_fc = nn.Linear(in_features, hidden_dim)
        
        # Camada Temporal (LSTM) que processará a sequência agregada pelo grafo
        self.lstm = nn.LSTM(hidden_dim * num_nodes, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape esperado: (Batch, Window, Features) -> Features serão interpretadas como Nodes
        batch_size, window_size, num_nodes = x.shape
        
        # Redimensiona para aplicar a convolução em grafos por timestep
        # (Batch * Window, Nodes, 1 feature por nó)
        x_g = x.view(-1, num_nodes, 1)
        
        # Propagação de mensagens no Grafo: A_norm @ X
        # self.A_norm shape: (Nodes, Nodes)
        support = torch.matmul(self.A_norm, x_g) # (Batch*Window, Nodes, 1)
        graph_out = F.relu(self.graph_fc(support)) # (Batch*Window, Nodes, Hidden_Dim)
        
        # Achata os nós para passar para a camada temporal
        # (Batch, Window, Nodes * Hidden_Dim)
        graph_out = graph_out.view(batch_size, window_size, num_nodes * graph_out.size(-1))
        
        # Processamento Temporal
        lstm_out, _ = self.lstm(graph_out)
        
        # Predição final baseada no último passo temporal da janela
        out = self.fc_out(lstm_out[:, -1, :])
        return out    

class SpatioTemporalDirectedGNN(nn.Module):
    def __init__(self, num_nodes, in_features=1, hidden_dim=64, output_dim=15):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. CONSTRUÇÃO DA MATRIZ DE ADJACÊNCIA DIRECIONADA CAUSAL
        # Fluxo estrito: Nó i -> Nó i+1 (SPO1 -> Ampli1 -> Ampli2 -> Ampli3 -> Ampli4 -> SPO2)
        adj_directed = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes - 1):
            adj_directed[i, i+1] = 1.0  # Conexão direcionada de montante para jusante
            
        # Adiciona Self-Loops para manter a identidade do próprio nó durante a propagação
        adj_directed = adj_directed + torch.eye(num_nodes)
        
        # Normalização assimétrica para Grafos Direcionados (Inversa da Matriz de Grau de Entrada)
        in_degree = torch.sum(adj_directed, dim=0)
        in_degree_inv = 1.0 / in_degree
        in_degree_inv[torch.isinf(in_degree_inv)] = 0.0
        D_in_inv = torch.diag(in_degree_inv)
        
        # Matriz de transição espacial direcionada normalizada: D_in^(-1) * A
        self.register_buffer('A_directed_norm', D_in_inv @ adj_directed)
        
        # Camadas da Rede
        self.graph_conv_weight = nn.Linear(in_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * num_nodes, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch, Window, Nodes)
        batch_size, window_size, num_nodes = x.shape
        
        # Redimensiona para processar o grafo por timestep: (Batch * Window, Nodes, 1)
        x_g = x.view(-1, num_nodes, 1)
        
        # Propagação Espacial Direcionada Causal
        # A_directed_norm garante que o sinal flua estritamente no sentido físico da cascata
        spatial_support = torch.matmul(self.A_directed_norm, x_g) 
        spatial_out = F.relu(self.graph_conv_weight(spatial_support)) # (Batch*Window, Nodes, Hidden_Dim)
        
        # Concatena a representação espacial dos nós para o processador temporal
        spatial_out = spatial_out.view(batch_size, window_size, num_nodes * self.hidden_dim)
        
        # Processamento Temporal da Propagação
        lstm_out, _ = self.lstm(spatial_out)
        
        # Predição baseada no último estado da janela de tempo
        out = self.fc_out(lstm_out[:, -1, :])
        return out   
    

    # Adicionar no final de src/models.py

class MultiTaskFaultDiagnosisGNN(nn.Module):
    def __init__(self, num_nodes, in_features=1, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. ENCODER ESPAÇO-TEMPORAL DIRECIONADO (Compartilhado)
        adj_directed = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes - 1):
            adj_directed[i, i+1] = 1.0
        adj_directed = adj_directed + torch.eye(num_nodes)
        
        in_degree = torch.sum(adj_directed, dim=0)
        in_degree_inv = 1.0 / in_degree
        in_degree_inv[torch.isinf(in_degree_inv)] = 0.0
        D_in_inv = torch.diag(in_degree_inv)
        
        self.register_buffer('A_directed_norm', D_in_inv @ adj_directed)
        self.graph_conv_weight = nn.Linear(in_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * num_nodes, hidden_dim, batch_first=True)
        
        # 2. CABEÇAS DE DIAGNÓSTICO (Multi-Task Heads)
        # Tarefa 1: Detecção (Binária: Saudável [0] ou Falha [1])
        self.head_detection = nn.Linear(hidden_dim, 1)
        
        # Tarefa 2: Localização (Multiclasse: Qual nó falhou? + Opção 0 para "Nenhum")
        # Se monitoramos 15 variáveis, temos 15 nós possíveis de origem da falha + 1 classe estável
        self.head_localization = nn.Linear(hidden_dim, num_nodes + 1)
        
        # Tarefa 3: Severidade (Multiclasse: 0 = Normal, 1 = Soft Failure, 2 = Hard Failure)
        self.head_severity = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        batch_size, window_size, num_nodes = x.shape
        
        # Extração de Features Geométricas e Direcionadas
        x_g = x.view(-1, num_nodes, 1)
        spatial_support = torch.matmul(self.A_directed_norm, x_g) 
        spatial_out = F.relu(self.graph_conv_weight(spatial_support))
        
        # Colapso temporal via LSTM
        spatial_out = spatial_out.view(batch_size, window_size, num_nodes * self.hidden_dim)
        lstm_out, _ = self.lstm(spatial_out)
        shared_features = lstm_out[:, -1, :] # Representação latente unificada do enlace
        
        # Distribuição para as cabeças especialistas (retorna Logits para estabilidade numérica nas perdas)
        out_detection = self.head_detection(shared_features)            # Shape: (Batch, 1)
        out_localization = self.head_localization(shared_features)      # Shape: (Batch, Num_Nodes + 1)
        out_severity = self.head_severity(shared_features)              # Shape: (Batch, 3)
        
        return out_detection, out_localization, out_severity