"""
DCell (Guo et al., SIGCOMM 2008) - implementação didática.

Cobre:
  - Estrutura:  t_k, g_k, endereçamento (k+1)-tupla <-> uid, BuildDCells
  - Roteamento: DCellRouting (Figura 3 do artigo)
  - Falhas:     local-reroute simplificado (apenas UM link falho, sem loops)
  - Verificações contra o artigo (exemplo de 7 saltos, Teorema 3, Tabela 1)

Convenções:
  - Endereço de servidor = tupla (a_k, ..., a_1, a_0), igual ao artigo.
  - Dois servidores do mesmo DCell0 contam como 1 salto (via mini-switch),
    como o artigo faz ao contar comprimento de caminho.
  - Link de nível i (i >= 1) liga dois servidores diretamente.

Uso:  python dcell.py
"""

import random
from collections import deque
from itertools import combinations


class DCell:
    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        # t[i] = servidores em um DCell_i ; g[i] = nº de DCell_{i-1} em um DCell_i
        self.t = [n]
        self.g = [1]
        for i in range(1, k + 1):
            self.g.append(self.t[i - 1] + 1)
            self.t.append(self.g[i] * self.t[i - 1])
        self.total = self.t[k]
        self._adj = None

    # ------------------------------------------------------------------
    # Endereçamento
    # ------------------------------------------------------------------
    def sub_addr(self, uid: int, level: int) -> tuple:
        """uid em [0, t_level) -> tupla (a_level, ..., a_0)."""
        a = []
        for j in range(level, 0, -1):
            a.append(uid // self.t[j - 1])
            uid %= self.t[j - 1]
        a.append(uid)
        return tuple(a)

    def addr_to_uid(self, addr: tuple) -> int:
        """uid_k = a_0 + sum_{j=1..k} a_j * t_{j-1}."""
        k = len(addr) - 1
        uid = addr[-1]
        for j in range(1, k + 1):
            uid += addr[k - j] * self.t[j - 1]
        return uid

    def uid_to_addr(self, uid: int) -> tuple:
        return self.sub_addr(uid, self.k)

    # ------------------------------------------------------------------
    # BuildDCells (Figura 2): gera a lista de links de nível >= 1
    # ------------------------------------------------------------------
    def build_links(self):
        """Retorna lista de (addr_u, addr_v, nivel). Segue o pseudocódigo."""
        links = []

        def build(pref: tuple, l: int):
            if l == 0:
                return  # Parte I: servidores ligados ao mini-switch (clique implícito)
            for i in range(self.g[l]):  # Parte II: constrói os DCell_{l-1}
                build(pref + (i,), l - 1)
            for i in range(self.t[l - 1]):  # Parte III: interliga os DCell_{l-1}
                for j in range(i + 1, self.g[l]):
                    n1 = pref + (i,) + self.sub_addr(j - 1, l - 1)
                    n2 = pref + (j,) + self.sub_addr(i, l - 1)
                    links.append((n1, n2, l))

        build((), self.k)
        return links

    def adjacency(self):
        """Grafo de servidores: clique nos DCell0 (via switch) + links de nível >= 1."""
        if self._adj is not None:
            return self._adj
        adj = [[] for _ in range(self.total)]
        # nível 0: servidores do mesmo DCell0 são vizinhos (1 salto via switch)
        for base in range(0, self.total, self.n):
            for u, v in combinations(range(base, base + self.n), 2):
                adj[u].append(v)
                adj[v].append(u)
        for a, b, _ in self.build_links():
            u, v = self.addr_to_uid(a), self.addr_to_uid(b)
            adj[u].append(v)
            adj[v].append(u)
        self._adj = adj
        return adj

    # ------------------------------------------------------------------
    # DCellRouting (Figura 3)
    # ------------------------------------------------------------------
    def get_link(self, pref: tuple, s: int, d: int):
        """Link que interliga os sub-DCells s e d (s != d) dentro do DCell com prefixo pref.
        Retorna (n1, n2): n1 pertence ao sub-DCell s e n2 ao sub-DCell d."""
        l = self.k - len(pref)
        if s < d:
            n1 = pref + (s,) + self.sub_addr(d - 1, l - 1)
            n2 = pref + (d,) + self.sub_addr(s, l - 1)
        else:
            n1 = pref + (s,) + self.sub_addr(d, l - 1)
            n2 = pref + (d,) + self.sub_addr(s - 1, l - 1)
        return n1, n2

    @staticmethod
    def common_prefix_len(a: tuple, b: tuple) -> int:
        m = 0
        for x, y in zip(a, b):
            if x != y:
                break
            m += 1
        return m

    def route(self, src: tuple, dst: tuple) -> list:
        """Caminho (lista de endereços) de src a dst, sem falhas."""
        if src == dst:
            return [src]
        m = self.common_prefix_len(src, dst)
        if m == self.k:  # mesmo DCell0
            return [src, dst]
        pref = src[:m]
        n1, n2 = self.get_link(pref, src[m], dst[m])
        return self.route(src, n1) + self.route(n2, dst)

    # ------------------------------------------------------------------
    # Local-reroute simplificado (um único link falho)
    # ------------------------------------------------------------------
    def route_with_failed_link(self, src: tuple, dst: tuple, failed: tuple) -> list:
        """Se o link 'failed' (par de endereços) estiver no caminho do DCellRouting,
        faz local-reroute: escolhe como proxy um link do MESMO nível que leve a outro
        sub-DCell, vai até o proxy e roteia dali até dst.
        Simplificações: sem retry/TTL, sem jump-up, sem link-state, uma só falha."""
        path = self.route(src, dst)
        bad = frozenset(failed)
        for idx, (a, b) in enumerate(zip(path, path[1:])):
            if frozenset((a, b)) == bad:
                break
        else:
            return path  # o link falho não está no caminho

        n1, n2 = path[idx], path[idx + 1]
        m = self.common_prefix_len(n1, n2)
        l = self.k - m
        if l == 0:
            raise ValueError("falha em link de nível 0 (mini-switch) não é tratada aqui")
        pref, s, d = n1[:m], n1[m], n2[m]
        c = next(x for x in range(self.g[l]) if x not in (s, d))  # sub-DCell alternativo
        p1, p2 = self.get_link(pref, s, c)  # p2 = proxy
        return path[: idx + 1] + self.route(n1, p1)[1:] + self.route(p2, dst)

    # ------------------------------------------------------------------
    # Utilidades de verificação
    # ------------------------------------------------------------------
    def bfs(self, src_uid: int) -> list:
        adj = self.adjacency()
        dist = [-1] * self.total
        dist[src_uid] = 0
        q = deque([src_uid])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def is_valid_path(self, path: list) -> bool:
        adj = self.adjacency()
        uids = [self.addr_to_uid(p) for p in path]
        return all(v in adj[u] for u, v in zip(uids, uids[1:]))


def fmt(path):
    return " -> ".join("[" + ",".join(map(str, p)) + "]" for p in path)


# ----------------------------------------------------------------------
# Demonstrações / verificações
# ----------------------------------------------------------------------
def demo_estrutura():
    print("=== Teorema 1: tamanho de um DCell_k ===")
    for n, k in [(4, 0), (4, 1), (4, 2), (4, 3), (6, 3)]:
        d = DCell(n, k)
        lo = (n + 0.5) ** (2 ** k) - 0.5
        hi = (n + 1) ** (2 ** k) - 1
        ok = (lo < d.total < hi) if k > 0 else True
        print(f"n={n} k={k}: t_k={d.total:>9,}  (limites: {lo:,.0f} < t_k < {hi:,.0f})  {'OK' if ok else 'FALHOU'}")

    print("\n=== DCell1 com n=4: os 10 cabos de nível 1 (Figura 1) ===")
    d = DCell(4, 1)
    for a, b, lvl in d.build_links():
        print(f"  nível {lvl}: [{a[0]},{a[1]}] <-> [{b[0]},{b[1]}]")
    degs = [len(x) for x in d.adjacency()]
    # grau no grafo = (n-1 vizinhos via switch) + 1 link nível 1
    print(f"  grau (no grafo de servidores): {set(degs)}  (esperado {{n-1+1}} = {{{4}}})")


def demo_exemplo_artigo():
    print("\n=== Exemplo do artigo: DCell2, n=2, [0,2,1] -> [1,2,1] ===")
    d = DCell(2, 2)
    src, dst = (0, 2, 1), (1, 2, 1)
    p = d.route(src, dst)
    print(f"DCellRouting ({len(p)-1} saltos): {fmt(p)}")
    assert d.is_valid_path(p) and len(p) - 1 == 7
    dist = d.bfs(d.addr_to_uid(src))[d.addr_to_uid(dst)]
    print(f"Caminho mínimo (BFS): {dist} saltos")
    assert dist == 3

    print("\n=== Com o link nível 2 [0,0,0]<->[1,0,0] falho (local-reroute) ===")
    q = d.route_with_failed_link(src, dst, ((0, 0, 0), (1, 0, 0)))
    print(f"Caminho com desvio ({len(q)-1} saltos): {fmt(q)}")
    assert d.is_valid_path(q) and len(q) - 1 == 9
    bad = frozenset(((0, 0, 0), (1, 0, 0)))
    assert all(frozenset(e) != bad for e in zip(q, q[1:])), "o link falho foi usado!"


def estatisticas(n, k, fontes=None, seed=0):
    """Média/desvio dos comprimentos: caminho mínimo vs DCellRouting (como a Tabela 1)."""
    d = DCell(n, k)
    rng = random.Random(seed)
    srcs = range(d.total) if fontes is None else rng.sample(range(d.total), fontes)
    bound = 2 ** (k + 1) - 1
    sp, dr = [], []
    for s in srcs:
        dist = d.bfs(s)
        sa = d.uid_to_addr(s)
        for t in range(d.total):
            if t == s:
                continue
            p = d.route(sa, d.uid_to_addr(t))
            assert len(p) - 1 <= bound, "violou o Teorema 3"
            if fontes is None or t % 7 == 0:  # nas amostras, valida só parte dos caminhos
                assert d.is_valid_path(p)
            sp.append(dist[t])
            dr.append(len(p) - 1)

    def ms(x):
        m = sum(x) / len(x)
        return m, (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5

    return d.total, ms(sp), ms(dr)


def demo_tabela1():
    print("\n=== Tabela 1 do artigo (média, desvio) ===")
    print("n k      t_k   shortest-path     DCellRouting     [artigo]")
    ref = {
        (4, 2): "4.87/1.27  5.16/1.42",
        (5, 2): "5.22/1.23  5.50/1.33",
        (6, 2): "5.48/1.18  5.73/1.25",
        (4, 3): "9.96/1.64  11.29/2.05",
    }
    for (n, k), r in ref.items():
        fontes = None if k == 2 else 6  # k=3: amostra de 6 origens (grafo grande)
        total, (m1, s1), (m2, s2) = estatisticas(n, k, fontes)
        tag = "" if fontes is None else "  (amostra)"
        print(f"{n} {k} {total:>8,}   {m1:5.2f}/{s1:4.2f}      {m2:5.2f}/{s2:4.2f}      [{r}]{tag}")


if __name__ == "__main__":
    demo_estrutura()
    demo_exemplo_artigo()
    demo_tabela1()
    print("\nTodas as verificações passaram.")
