from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from collections import deque, defaultdict
import random

# ============================ Configuración básica ============================
DEFAULT_EPS = "ε"     # configurable con --eps
OTHER = "OTHER"       # símbolo sintético para "cualquier otro" (se dibuja como ANY)


# ============================ Tokenización / ER ===============================
@dataclass
class Token:
    kind: str                 # 'LIT','ANY','CLASS','LP','RP','ALT','STAR','PLUS','QMARK','CONCAT','REPEAT'
    value: Optional[str] = None


def tokenize(expr: str) -> List[Token]:
    """
    Tokeniza respetando:
      - Escapes: '\\x' => LIT('x')
      - Clases:  '[ ... ]' => CLASS(contenido crudo)
      - Punto:   '.' => ANY (cualquier carácter); '\\.' => LIT('.')
      - Operadores: '(', ')', '|', '*', '+', '?'
      - Repetición: '{m}', '{m,}', '{m,n}' => REPEAT('{...}')
    Ignora espacios en blanco.
    """
    tokens: List[Token] = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c == "\\":  # escape
            if i + 1 < n:
                tokens.append(Token("LIT", expr[i + 1]))
                i += 2
            else:
                tokens.append(Token("LIT", "\\"))
                i += 1
        elif c == "[":  # clase de caracteres
            j = i + 1
            contenido = ""
            cerrado = False
            while j < n:
                if expr[j] == "\\" and j + 1 < n:
                    contenido += "\\" + expr[j + 1]
                    j += 2
                    continue
                if expr[j] == "]":
                    cerrado = True
                    break
                contenido += expr[j]
                j += 1
            if cerrado:
                tokens.append(Token("CLASS", contenido))
                i = j + 1
            else:
                tokens.append(Token("LIT", "["))
                i += 1
        elif c == ".":
            tokens.append(Token("ANY", "."))
            i += 1
        elif c in "()*+?|":
            kind_map = {"(":"LP", ")":"RP", "*":"STAR", "+":"PLUS", "?":"QMARK", "|":"ALT"}
            tokens.append(Token(kind_map[c], c))
            i += 1
        elif c == "{":  # cuantificadores
            j = i + 1
            interior = ""
            valido = False
            while j < n and expr[j] != "}":
                interior += expr[j]
                j += 1
            if j < n and expr[j] == "}":
                s = interior.strip()
                if s and all(ch.isdigit() or ch == "," for ch in s):
                    tokens.append(Token("REPEAT", "{" + s + "}"))
                    i = j + 1
                    valido = True
            if not valido:
                tokens.append(Token("LIT", "{"))
                i += 1
        elif c.isspace():
            i += 1
        else:
            tokens.append(Token("LIT", c))
            i += 1
    return tokens


def render_atom(tok: Token) -> str:
    if tok.kind == "ANY":
        return "ANY"
    if tok.kind == "CLASS":
        return f"[{tok.value}]"
    if tok.kind == "LIT" and tok.value == ".":
        return r"\."
    return tok.value


def needs_concat(prev: Token, nxt: Token) -> bool:
    starts_atom = nxt.kind in ("LIT", "ANY", "CLASS", "LP")
    ends_atom   = prev.kind in ("LIT", "ANY", "CLASS", "RP", "STAR", "PLUS", "QMARK", "REPEAT")
    if prev.kind in ("LP", "ALT"):
        return False
    if nxt.kind in ("RP", "ALT"):
        return False
    return ends_atom and starts_atom


def add_concat(tokens: List[Token]) -> List[Token]:
    res: List[Token] = []
    for t in tokens:
        if res and needs_concat(res[-1], t):
            res.append(Token("CONCAT", "·"))
        res.append(t)
    return res


# =============================== Shunting Yard ===============================
PRECEDENCE = {"ALT": 1, "CONCAT": 2}

@dataclass
class Step:
    i: int
    accion: str
    token: str
    salida: str
    pila: str


def shunting_yard(tokens: List[Token]) -> Tuple[List[str], List[Step]]:
    out: List[str] = []
    ops: List[Token] = []
    pasos: List[Step] = []

    for i, tok in enumerate(tokens):
        if tok.kind in ("LIT", "ANY", "CLASS"):
            atom = render_atom(tok)
            out.append(atom)
            pasos.append(Step(i, "EMIT", atom, " ".join(out), "".join(o.value or o.kind for o in ops)))
        elif tok.kind in ("STAR", "PLUS", "QMARK", "REPEAT"):
            out.append(tok.value or {"STAR":"*", "PLUS":"+", "QMARK":"?", "REPEAT":"{?}"}[tok.kind])
            pasos.append(Step(i, "EMIT_POSF", tok.value or tok.kind, " ".join(out), "".join(o.value or o.kind for o in ops)))
        elif tok.kind == "LP":
            ops.append(tok)
            pasos.append(Step(i, "PUSH", "(", " ".join(out), "".join(o.value or o.kind for o in ops)))
        elif tok.kind == "RP":
            while ops and ops[-1].kind != "LP":
                p = ops.pop()
                out.append(p.value or p.kind)
                pasos.append(Step(i, "POP->EMIT", p.value or p.kind, " ".join(out), "".join(o.value or o.kind for o in ops)))
            if not ops:
                raise ValueError("Paréntesis desbalanceados (falta '(')")
            ops.pop()
            pasos.append(Step(i, "POP", ")", " ".join(out), "".join(o.value or o.kind for o in ops)))
        elif tok.kind in ("ALT", "CONCAT"):
            while ops and ops[-1].kind in ("ALT", "CONCAT") and PRECEDENCE[ops[-1].kind] >= PRECEDENCE[tok.kind]:
                p = ops.pop()
                out.append(p.value or p.kind)
                pasos.append(Step(i, "POP->EMIT", p.value or p.kind, " ".join(out), "".join(o.value or o.kind for o in ops)))
            ops.append(tok)
            pasos.append(Step(i, "PUSH", tok.value or tok.kind, " ".join(out), "".join(o.value or o.kind for o in ops)))
        else:
            out.append(tok.value or tok.kind)
            pasos.append(Step(i, "EMIT(?)", tok.value or tok.kind, " ".join(out), "".join(o.value or o.kind for o in ops)))

    while ops:
        p = ops.pop()
        if p.kind == "LP":
            raise ValueError("Paréntesis desbalanceados al final")
        out.append(p.value or p.kind)
        pasos.append(Step(len(tokens), "POP->EMIT", p.value or p.kind, " ".join(out), "".join(o.value or o.kind for o in ops)))
    return out, pasos


# ===================== Expansión canónica de cuantificadores =================
def _repeat_expr(expr_tokens: List[str], k: int) -> List[str]:
    if k == 0:
        return ["ε"]
    out = expr_tokens[:]
    for _ in range(1, k):
        out = out + expr_tokens + ["·"]
    return out


def expand_postfix(postfix: List[str]) -> List[str]:
    st: List[List[str]] = []
    for t in postfix:
        if t == "*":
            a = st.pop()
            st.append(a + ["*"])
        elif t == "+":
            a = st.pop()
            st.append(a + a + ["*", "·"])
        elif t == "?":
            a = st.pop()
            st.append(a + ["ε", "|"])
        elif t == "·":
            b = st.pop(); a = st.pop()
            st.append(a + b + ["·"])
        elif t == "|":
            b = st.pop(); a = st.pop()
            st.append(a + b + ["|"])
        elif t.startswith("{") and t.endswith("}"):
            a = st.pop()
            parts = t[1:-1].split(",")
            if len(parts) == 1:
                m = int(parts[0])
                st.append(_repeat_expr(a, m))
            elif parts[1] == "":
                m = int(parts[0])
                st.append(_repeat_expr(a, m) + a + ["*", "·"])
            else:
                m = int(parts[0]); n = int(parts[1])
                alts: List[str] = []
                for k in range(m, n + 1):
                    ek = _repeat_expr(a, k)
                    if not alts:
                        alts = ek
                    else:
                        alts = alts + ek + ["|"]
                st.append(alts)
        else:
            st.append([t])
    if len(st) != 1:
        raise ValueError("Postfix inválido (sobraron elementos)")
    return st[0]


# =============================== Thompson AFN ================================
@dataclass(eq=False)
class State:
    id: int
    edges: Dict[str, Set["State"]]
    def __hash__(self) -> int:
        return hash(self.id)


class NFA:
    def __init__(self, eps: str = DEFAULT_EPS):
        self._next_id = 0
        self.start: Optional[State] = None
        self.accept: Optional[State] = None
        self.states: Set[State] = set()
        self.eps = eps

    def _new_state(self) -> State:
        s = State(self._next_id, {})
        self._next_id += 1
        self.states.add(s)
        return s

    def _add_edge(self, u: State, sym: str, v: State):
        u.edges.setdefault(sym, set()).add(v)

    def _lit(self, symbol: str) -> Tuple[State, State]:
        s, t = self._new_state(), self._new_state()
        if symbol == "ε":
            self._add_edge(s, self.eps, t)
        elif symbol.startswith("[") and symbol.endswith("]"):
            for ch in expand_class(symbol[1:-1]):
                self._add_edge(s, ch, t)
        elif symbol == "ANY":
            self._add_edge(s, "ANY", t)
        else:
            self._add_edge(s, symbol, t)
        return s, t

    def _concat(self, a: Tuple[State, State], b: Tuple[State, State]) -> Tuple[State, State]:
        self._add_edge(a[1], self.eps, b[0])
        return a[0], b[1]

    def _alt(self, a: Tuple[State, State], b: Tuple[State, State]) -> Tuple[State, State]:
        s, t = self._new_state(), self._new_state()
        self._add_edge(s, self.eps, a[0]); self._add_edge(s, self.eps, b[0])
        self._add_edge(a[1], self.eps, t); self._add_edge(b[1], self.eps, t)
        return s, t

    def _star(self, a: Tuple[State, State]) -> Tuple[State, State]:
        s, t = self._new_state(), self._new_state()
        self._add_edge(s, self.eps, a[0]); self._add_edge(s, self.eps, t)
        self._add_edge(a[1], self.eps, a[0]); self._add_edge(a[1], self.eps, t)
        return s, t

    def build_from_postfix(self, pf: List[str]):
        st: List[Tuple[State, State]] = []
        for tok in pf:
            if tok == "·":
                b = st.pop(); a = st.pop()
                st.append(self._concat(a, b))
            elif tok == "|":
                b = st.pop(); a = st.pop()
                st.append(self._alt(a, b))
            elif tok == "*":
                a = st.pop(); st.append(self._star(a))
            else:
                st.append(self._lit(tok))
        if len(st) != 1:
            raise ValueError("Postfix inválido para Thompson")
        self.start, self.accept = st[0]

    def _eps_closure(self, S: Set[State]) -> Set[State]:
        stack = list(S); seen = set(S)
        while stack:
            u = stack.pop()
            for v in u.edges.get(self.eps, ()):
                if v not in seen:
                    seen.add(v); stack.append(v)
        return seen

    def _move(self, S: Set[State], sym: str) -> Set[State]:
        out: Set[State] = set()
        for u in S:
            for v in u.edges.get(sym, ()):
                out.add(v)
            for v in u.edges.get("ANY", ()):
                out.add(v)
        return out

    def accepts(self, w: str) -> bool:
        if self.start is None or self.accept is None:
            return False
        current = self._eps_closure({self.start})
        for ch in w:
            current = self._eps_closure(self._move(current, ch))
            if not current:
                break
        return self.accept in current


# ============================== Utilidades varias ============================
def expand_class(payload: str) -> Set[str]:
    out: Set[str] = set()
    i = 0
    L = len(payload)
    while i < L:
        c = payload[i]
        if c == "\\" and i + 1 < L:
            out.add(payload[i + 1]); i += 2; continue
        if i + 2 < L and payload[i + 1] == "-":
            a, b = payload[i], payload[i + 2]
            lo, hi = (ord(a), ord(b)) if ord(a) <= ord(b) else (ord(b), ord(a))
            for code in range(lo, hi + 1):
                out.add(chr(code))
            i += 3
        else:
            out.add(c); i += 1
    return out


# ============================== Dibujo AFN ===================================
def layout_positions_nfa(nfa: NFA) -> Dict[State, Tuple[float, float]]:
    adj: Dict[State, Set[State]] = {s: set() for s in nfa.states}
    for s in nfa.states:
        for dests in s.edges.values():
            adj[s] |= dests

    dist: Dict[State, int] = {}
    if nfa.start is not None:
        dist[nfa.start] = 0
        q = deque([nfa.start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
    maxd = max(dist.values()) if dist else 0
    for s in nfa.states:
        dist.setdefault(s, maxd)

    levels: Dict[int, List[State]] = {}
    for s, d in dist.items():
        levels.setdefault(d, []).append(s)

    pos: Dict[State, Tuple[float, float]] = {}
    sep_x, sep_y = 2.4, 1.6
    for d in sorted(levels):
        ys = sorted(levels[d], key=lambda z: z.id)
        n = len(ys)
        for i, s in enumerate(ys):
            y = (i - (n - 1) / 2.0) * sep_y
            x = d * sep_x
            pos[s] = (x, y)
    return pos


def draw_nfa_png(nfa: NFA, filename_png: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Arc

    pos = layout_positions_nfa(nfa)
    xs = [p[0] for p in pos.values()] + [-2]
    ys = [p[1] for p in pos.values()] + [0]
    x_min, x_max = min(xs) - 1.2, max(xs) + 1.2
    y_min, y_max = min(ys) - 1.2, max(ys) + 1.2

    fig, ax = plt.subplots(figsize=(max(6, (x_max - x_min) * 1.2),
                                    max(4, (y_max - y_min) * 1.2)))
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.axis("off")

    R = 0.28

    for s, (x, y) in pos.items():
        circ = Circle((x, y), R, fill=False, lw=2)
        ax.add_patch(circ)
        if nfa.accept is not None and s.id == nfa.accept.id:
            circ2 = Circle((x, y), R - 0.06, fill=False, lw=2)
            ax.add_patch(circ2)
        ax.text(x, y, str(s.id), ha="center", va="center", fontsize=10)

    if nfa.start is not None:
        x0, y0 = pos[nfa.start]
        arr = FancyArrowPatch((-2, y0), (x0 - R, y0), arrowstyle="->", lw=1.6, mutation_scale=14)
        ax.add_patch(arr)

    for s, (x1, y1) in pos.items():
        for sym, dests in s.edges.items():
            for t in dests:
                x2, y2 = pos[t]
                if s is t:
                    arc = Arc((x1, y1 + R + 0.20), 0.8, 0.6, angle=0, theta1=220, theta2=-40, lw=1.5)
                    ax.add_patch(arc)
                    ax.text(x1 + 0.05, y1 + R + 0.7, sym, fontsize=9)
                    continue
                arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", lw=1.5,
                                      mutation_scale=12, shrinkA=12, shrinkB=12)
                ax.add_patch(arr)
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0 + 0.15
                ax.text(xm, ym, sym, fontsize=9)

    fig.tight_layout()
    if not filename_png.lower().endswith(".png"):
        filename_png += ".png"
    fig.savefig(filename_png, dpi=150)
    plt.close(fig)


# ============================== Front–End Parte A ============================
def regex_to_postfix(expr: str) -> Tuple[List[str], List[Step]]:
    toks = add_concat(tokenize(expr))
    postfix, steps = shunting_yard(toks)
    canonical = expand_postfix(postfix)
    return canonical, steps


def build_nfa_from_regex(expr: str, eps: str = DEFAULT_EPS) -> Tuple[NFA, List[str]]:
    pf, _steps = regex_to_postfix(expr)
    nfa = NFA(eps=eps)
    nfa.build_from_postfix(pf)
    return nfa, pf


# ======================= Subconjuntos: AFN -> AFD ============================
@dataclass(eq=False)
class DfaState:
    id: int
    trans: Dict[str, int]
    accept: bool
    def __hash__(self) -> int:
        return hash(self.id)


class DFA:
    def __init__(self, alphabet: Set[str]):
        self.states: Dict[int, DfaState] = {}
        self.start: Optional[int] = None
        self.alphabet: Set[str] = set(alphabet)  # puede incluir OTHER

    def add_state(self, accept: bool) -> int:
        i = len(self.states)
        self.states[i] = DfaState(i, {}, accept)
        if self.start is None:
            self.start = i
        return i

    def add_trans(self, u: int, sym: str, v: int):
        self.states[u].trans[sym] = v

    def accepts(self, w: str) -> bool:
        if self.start is None:
            return False
        cur = self.start
        for ch in w:
            s = self.states[cur]
            if ch in s.trans:
                cur = s.trans[ch]
            elif OTHER in s.trans:
                cur = s.trans[OTHER]
            else:
                return False
        return self.states[cur].accept

    def complete_with_sink(self) -> int:
        sink = None
        for st in self.states.values():
            for a in self.alphabet:
                if a not in st.trans:
                    if sink is None:
                        sink = self.add_state(False)
                    st.trans[a] = sink
        if sink is None:
            sink = self.add_state(False)
        for a in self.alphabet:
            self.states[sink].trans[a] = sink
        return sink


def nfa_alphabet(nfa: NFA) -> Set[str]:
    Σ: Set[str] = set()
    for s in nfa.states:
        for a in s.edges.keys():
            if a == nfa.eps or a == "ANY":
                continue
            Σ.add(a)
    return Σ


def eclosure(nfa: NFA, S: Set[State]) -> Set[State]:
    return nfa._eps_closure(S)


def move_set(nfa: NFA, S: Set[State], sym: str) -> Set[State]:
    out = set()
    for u in S:
        out |= u.edges.get(sym, set())
    return out


def build_dfa_from_nfa(nfa: NFA) -> Tuple[DFA, Dict[FrozenSet[State], int]]:
    Σ = nfa_alphabet(nfa)
    has_any = any("ANY" in s.edges for s in nfa.states)
    alphabet = set(Σ)
    if has_any:
        alphabet.add(OTHER)

    dfa = DFA(alphabet=alphabet)
    subset_id: Dict[FrozenSet[State], int] = {}

    start_set = eclosure(nfa, {nfa.start})
    start_key = frozenset(start_set)
    start_id = dfa.add_state(accept=(nfa.accept in start_set))
    subset_id[start_key] = start_id

    Q = deque([start_key])
    while Q:
        T = Q.popleft()
        u = subset_id[T]
        for a in Σ:
            U_raw = move_set(nfa, T, a)
            if has_any:
                U_raw |= move_set(nfa, T, "ANY")
            U = eclosure(nfa, U_raw)
            key = frozenset(U)
            if key not in subset_id:
                qid = dfa.add_state(accept=(nfa.accept in U))
                subset_id[key] = qid
                Q.append(key)
            dfa.add_trans(u, a, subset_id[key])
        if has_any:
            U_raw = move_set(nfa, T, "ANY")
            U = eclosure(nfa, U_raw)
            key = frozenset(U)
            if key not in subset_id:
                qid = dfa.add_state(accept=(nfa.accept in U))
                subset_id[key] = qid
                Q.append(key)
            dfa.add_trans(u, OTHER, subset_id[key])
    return dfa, subset_id


# =============================== Minimización ================================
def clone_dfa(dfa: DFA) -> DFA:
    c = DFA(alphabet=set(dfa.alphabet))
    id_map = {}
    for sid in dfa.states:
        id_map[sid] = c.add_state(dfa.states[sid].accept)
    c.start = id_map[dfa.start]
    for u, st in dfa.states.items():
        for a, v in st.trans.items():
            c.add_trans(id_map[u], a, id_map[v])
    return c


def hopcroft_minimize(dfa: DFA) -> DFA:
    sink = dfa.complete_with_sink()
    Σ = sorted(dfa.alphabet)
    states = list(dfa.states.keys())
    F = {s.id for s in dfa.states.values() if s.accept}
    NF = set(dfa.states.keys()) - F

    P = [F, NF]
    W = [F.copy(), NF.copy()]
    trans = {s: dfa.states[s].trans.copy() for s in states}

    while W:
        A = W.pop()
        for a in Σ:
            X = {s for s in states if trans[s][a] in A}
            newP = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    newP.extend([inter, diff])
                    if Y in W:
                        W.remove(Y)
                        W.extend([inter, diff])
                    else:
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    newP.append(Y)
            P = newP

    block_id = {}
    for i, block in enumerate(P):
        for s in block:
            block_id[s] = i

    min_dfa = DFA(alphabet=set(dfa.alphabet))
    new_state_map = {}
    for i, block in enumerate(P):
        accept = any(dfa.states[s].accept for s in block)
        new_state_map[i] = min_dfa.add_state(accept)

    for i, block in enumerate(P):
        rep = next(iter(block))
        for a in dfa.alphabet:
            to_old = dfa.states[rep].trans[a]
            j = block_id[to_old]
            min_dfa.add_trans(new_state_map[i], a, new_state_map[j])

    start_block = block_id[dfa.start]
    min_dfa.start = new_state_map[start_block]
    return min_dfa


# ============================== Dibujo AFD ===================================
def layout_positions_dfa(dfa: DFA) -> Dict[int, Tuple[float, float]]:
    adj = {i: set(s.trans.values()) for i, s in dfa.states.items()}
    dist = {}
    if dfa.start is not None:
        dist[dfa.start] = 0
        q = deque([dfa.start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
    maxd = max(dist.values()) if dist else 0
    for s in dfa.states:
        dist.setdefault(s, maxd)
    levels = {}
    for s, d in dist.items():
        levels.setdefault(d, []).append(s)
    pos = {}
    sep_x, sep_y = 2.6, 1.6
    for d in sorted(levels):
        ys = sorted(levels[d])
        n = len(ys)
        for i, s in enumerate(ys):
            y = (i - (n - 1) / 2.0) * sep_y
            x = d * sep_x
            pos[s] = (x, y)
    return pos


def draw_dfa_png(dfa: DFA, filename_png: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Arc

    pos = layout_positions_dfa(dfa)
    xs = [p[0] for p in pos.values()] + [-2]
    ys = [p[1] for p in pos.values()] + [0]
    x_min, x_max = min(xs) - 1.2, max(xs) + 1.2
    y_min, y_max = min(ys) - 1.2, max(ys) + 1.2

    fig, ax = plt.subplots(figsize=(max(6, (x_max - x_min) * 1.2),
                                    max(4, (y_max - y_min) * 1.2)))
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.axis("off")
    R = 0.28

    labels = defaultdict(list)
    for u, s in dfa.states.items():
        circ = Circle((pos[u][0], pos[u][1]), R, fill=False, lw=2)
        ax.add_patch(circ)
        if s.accept:
            circ2 = Circle((pos[u][0], pos[u][1]), R - 0.06, fill=False, lw=2)
            ax.add_patch(circ2)
        ax.text(pos[u][0], pos[u][1], str(u), ha="center", va="center", fontsize=10)
        if dfa.start == u:
            arr = FancyArrowPatch((-2, pos[u][1]), (pos[u][0] - R, pos[u][1]), arrowstyle="->", lw=1.6, mutation_scale=14)
            ax.add_patch(arr)
        for a, v in s.trans.items():
            labels[(u, v)].append(a)

    for (u, v), syms in labels.items():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        if u == v:
            arc = Arc((x1, y1 + R + 0.20), 0.8, 0.6, angle=0, theta1=220, theta2=-40, lw=1.5)
            ax.add_patch(arc)
            lbl = ",".join("ANY" if s == OTHER else s for s in sorted(syms))
            ax.text(x1 + 0.05, y1 + R + 0.7, lbl, fontsize=9)
            continue
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", lw=1.5,
                              mutation_scale=12, shrinkA=12, shrinkB=12)
        ax.add_patch(arr)
        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0 + 0.15
        lbl = ",".join("ANY" if s == OTHER else s for s in sorted(syms))
        ax.text(xm, ym, lbl, fontsize=9)

    fig.tight_layout()
    if not filename_png.lower().endswith(".png"):
        filename_png += ".png"
    fig.savefig(filename_png, dpi=150)
    plt.close(fig)


# =============================== Verificación ================================
def dfa_equiv(d1: DFA, d2: DFA) -> bool:
    Σ = set(d1.alphabet) | set(d2.alphabet)
    d1c = clone_dfa(d1); d1c.alphabet = set(Σ); d1c.complete_with_sink()
    d2c = clone_dfa(d2); d2c.alphabet = set(Σ); d2c.complete_with_sink()

    start = (d1c.start, d2c.start)
    seen = set([start])
    Q = deque([start])
    while Q:
        u1, u2 = Q.popleft()
        if d1c.states[u1].accept != d2c.states[u2].accept:
            return False
        for a in Σ:
            v1 = d1c.states[u1].trans[a]
            v2 = d2c.states[u2].trans[a]
            p = (v1, v2)
            if p not in seen:
                seen.add(p)
                Q.append(p)
    return True


def random_words(Σ: Set[str], trials: int, max_len: int = 6, seed: int = 123) -> List[str]:
    rng = random.Random(seed)
    letters = sorted(Σ)
    out = []
    for _ in range(trials):
        L = rng.randint(0, max_len)
        if letters:
            w = "".join(rng.choice(letters) for _ in range(L))
        else:
            w = ""
        out.append(w)
    return out


def diff_test(nfa: NFA, dfa: DFA, trials: int = 200, max_len: int = 6, alphabet_hint: Optional[str] = None) -> Tuple[int, int]:
    Σ = set(nfa_alphabet(nfa))
    if alphabet_hint:
        Σ |= set(alphabet_hint)
    words = random_words(Σ, trials=trials, max_len=max_len)
    oks = 0; mism = 0
    for w in words:
        a = nfa.accepts(w)
        b = dfa.accepts(w)
        if a == b:
            oks += 1
        else:
            mism += 1
    return oks, mism


# ============================== CLI principal ================================
def main():
    ap = argparse.ArgumentParser(description="Proyecto completo: Parte A + Parte B (AFN, AFD, Minimización, PNGs y verificación)")
    ap.add_argument("--mode", choices=["A", "B", "ALL"], default="ALL", help="A: solo AFN; B: AFN→AFD+min; ALL: ambos")
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--regex", help="Expresión regular única (infix)")
    g_in.add_argument("--input", help="Archivo con ER (una por línea)")

    g_w = ap.add_mutually_exclusive_group()
    g_w.add_argument("--word", help="Cadena w única para todas las ER")
    g_w.add_argument("--words", help="Archivo con cadenas w (paralelo a --input)")

    ap.add_argument("--alphabet", help="Pista de alfabeto para test aleatorios, ej. 'ab'")
    ap.add_argument("--outdir", default="out_all", help="Carpeta de salida para PNGs")
    ap.add_argument("--eps", default=DEFAULT_EPS, help="Símbolo visible para ε (usado en los dibujos)")
    ap.add_argument("--show-steps", action="store_true", help="Imprime pasos del Shunting Yard (modo A/ALL)")

    ap.add_argument("--check", action="store_true", help="Verifica equivalencia DFA ≡ DFAmin (modo B/ALL)")
    ap.add_argument("--random", type=int, default=0, help="Corre pruebas aleatorias NFA↔DFA con N muestras (modo B/ALL)")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Cargar ERs
    if args.regex is not None:
        regexes = [args.regex.strip()]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            regexes = [ln.strip() for ln in f if ln.strip()]

    # Cargar w
    if args.word is not None:
        words = [args.word] * len(regexes)
    elif args.words is not None:
        with open(args.words, "r", encoding="utf-8") as f:
            words = [ln.rstrip("\n") for ln in f]
        if len(words) < len(regexes):
            words += [""] * (len(regexes) - len(words))
    else:
        words = [""] * len(regexes)

    for i, (reg, w) in enumerate(zip(regexes, words), 1):
        print("=" * 80)
        print(f"[{i}] ER: {reg}")
        pf, steps = regex_to_postfix(reg)
        print("POSTFIX (canónica):", " ".join(pf))

        # ----- Parte A -----
        nfa = NFA(eps=args.eps)
        nfa.build_from_postfix(pf)
        if args.show_steps:
            print("\nPasos Shunting-Yard:")
            for st in steps:
                print(f"  [{st.i:02d}] {st.accion:10s} {st.token:10s} OUT: {st.salida:30s} OPS: {st.pila}")
        if args.mode in ("A", "ALL"):
            nfa_png = os.path.join(args.outdir, f"afn_{i}.png")
            draw_nfa_png(nfa, nfa_png)
            print(f"Imagen AFN -> {nfa_png}")
            w0 = "" if w is None else w
            print(f"w = {repr(w0)}  =>  NFA: {'sí' if nfa.accepts(w0) else 'no'}")

        # ----- Parte B -----
        if args.mode in ("B", "ALL"):
            dfa, subset_map = build_dfa_from_nfa(nfa)
            dfa_min = hopcroft_minimize(clone_dfa(dfa))
            dfa_png = os.path.join(args.outdir, f"afd_{i}.png")
            dfa_min_png = os.path.join(args.outdir, f"afd_min_{i}.png")
            draw_dfa_png(dfa, dfa_png); draw_dfa_png(dfa_min, dfa_min_png)
            print(f"Imagen AFD     -> {dfa_png}")
            print(f"Imagen AFDmin  -> {dfa_min_png}")
            w0 = "" if w is None else w
            a_nfa = nfa.accepts(w0); a_dfa = dfa.accepts(w0); a_min = dfa_min.accepts(w0)
            print(f"w = {repr(w0)}  =>  NFA: {'sí' if a_nfa else 'no'} | DFA: {'sí' if a_dfa else 'no'} | DFAmin: {'sí' if a_min else 'no'}")
            if args.check:
                ok = dfa_equiv(dfa, dfa_min)
                print(f"Equivalencia DFA ≡ DFAmin: {'OK' if ok else 'FALLA'}")
            if args.random and args.random > 0:
                oks, mism = diff_test(nfa, dfa, trials=args.random, max_len=6, alphabet_hint=args.alphabet)
                print(f"Prueba NFA↔DFA aleatoria: OK={oks}  MISMATCH={mism}  (N={args.random})")
                if mism > 0:
                    print("⚠️ Hay discrepancias; revisar manejo de '.'/ANY o clases de caracteres.")

if __name__ == "__main__":
    main()
