import torch
import torch.nn as nn


class StackedRNNRegressor(nn.Module):
    """Vanilla RNN / LSTM stacked architecture as described in the paper:
    4 sequential recurrent hidden layers (100, 80, 60, 40 units) each with
    dropout=0.2, followed by 2 dense layers (3*m, 2*m units) and a final
    linear output layer projecting to m (number of time series)."""

    def __init__(self, m, cell="rnn", dropout=0.2):
        super().__init__()
        Cell = nn.RNN if cell == "rnn" else nn.LSTM
        self.cell_type = cell
        self.rnn1 = Cell(input_size=m, hidden_size=100, batch_first=True, nonlinearity="relu") if cell == "rnn" else Cell(input_size=m, hidden_size=100, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.rnn2 = Cell(input_size=100, hidden_size=80, batch_first=True, nonlinearity="relu") if cell == "rnn" else Cell(input_size=100, hidden_size=80, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.rnn3 = Cell(input_size=80, hidden_size=60, batch_first=True, nonlinearity="relu") if cell == "rnn" else Cell(input_size=80, hidden_size=60, batch_first=True)
        self.drop3 = nn.Dropout(dropout)
        self.rnn4 = Cell(input_size=60, hidden_size=40, batch_first=True, nonlinearity="relu") if cell == "rnn" else Cell(input_size=60, hidden_size=40, batch_first=True)
        self.drop4 = nn.Dropout(dropout)

        self.dense1 = nn.Linear(40, 3 * m)
        self.dense2 = nn.Linear(3 * m, 2 * m)
        self.out = nn.Linear(2 * m, m)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq, m)
        h, _ = self.rnn1(x)
        h = self.drop1(h)
        h, _ = self.rnn2(h)
        h = self.drop2(h)
        h, _ = self.rnn3(h)
        h = self.drop3(h)
        h, _ = self.rnn4(h)
        h = self.drop4(h)
        last = h[:, -1, :]  # last timestep
        d = self.act(self.dense1(last))
        d = self.act(self.dense2(d))
        return self.out(d)


class LSTNet(nn.Module):
    """LSTNet as described in Lai et al. 2018 / adapted by Silva et al. 2022:
    Conv1D -> {GRU, Recurrent-Skip} -> dense regression layer -> + AR component.
    """

    def __init__(self, m, window=5, conv_filters=10, kernel_size=3,
                 gru_units=3, skip_units=2, skip=2, ar_window=2):
        super().__init__()
        self.m = m
        self.window = window
        self.skip = skip
        self.ar_window = ar_window
        self.conv_filters = conv_filters

        # Conv1d over the time dimension, treating m as channels-in
        pad = 0
        self.conv = nn.Conv1d(in_channels=m, out_channels=conv_filters,
                               kernel_size=kernel_size, padding=pad)
        self.conv_act = nn.ReLU()
        conv_out_len = window - kernel_size + 1  # no padding, stride 1

        # GRU recurrent layer
        self.gru = nn.GRU(input_size=conv_filters, hidden_size=gru_units, batch_first=True)

        # Recurrent-skip layer: reshape conv output using skip step, run GRU over skip-strided sequence
        self.skip_gru = nn.GRU(input_size=conv_filters, hidden_size=skip_units, batch_first=True)
        self.conv_out_len = conv_out_len

        # Dense regression layer combining hr_t (gru_units) and p*hrs_t (skip*skip_units)
        self.p = max(1, conv_out_len // skip) if skip > 0 else 0
        dense_in = gru_units + (skip * skip_units if skip > 0 else 0)
        self.regression = nn.Linear(dense_in, m)

        # Autoregressive component (linear, per-series over ar_window last raw steps)
        self.ar_window = min(ar_window, window)
        self.ar = nn.Linear(self.ar_window, 1)  # applied per series independently

        self.out_act = nn.ReLU()
        self.out_layer = nn.Linear(m, m)

    def forward(self, x):
        # x: (batch, window, m)
        batch = x.size(0)
        c_in = x.permute(0, 2, 1)  # (batch, m, window) channels=m
        c = self.conv_act(self.conv(c_in))  # (batch, conv_filters, conv_out_len)
        c_seq = c.permute(0, 2, 1)  # (batch, conv_out_len, conv_filters)

        # GRU branch
        _, h_gru = self.gru(c_seq)
        h_r = h_gru[-1]  # (batch, gru_units)

        # Recurrent-skip branch: take last p*skip steps, reshape to (batch, p, skip, conv_filters)
        if self.p > 0 and self.skip > 0:
            usable_len = self.p * self.skip
            c_skip = c_seq[:, -usable_len:, :]  # (batch, usable_len, conv_filters)
            c_skip = c_skip.reshape(batch, self.p, self.skip, self.conv_filters)
            c_skip = c_skip.permute(0, 2, 1, 3).reshape(batch * self.skip, self.p, self.conv_filters)
            _, h_skip = self.skip_gru(c_skip)
            h_rs = h_skip[-1].reshape(batch, self.skip * h_skip.size(-1))
        else:
            h_rs = torch.zeros(batch, 0, device=x.device)

        h_d = torch.cat([h_r, h_rs], dim=1)
        neural_out = self.regression(h_d)  # (batch, m)

        # AR component: use last ar_window raw values of each series
        ar_in = x[:, -self.ar_window:, :]  # (batch, ar_window, m)
        ar_in = ar_in.permute(0, 2, 1)  # (batch, m, ar_window)
        ar_out = self.ar(ar_in).squeeze(-1)  # (batch, m)

        combined = neural_out + ar_out
        return self.out_layer(self.out_act(combined))
