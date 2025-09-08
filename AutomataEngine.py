from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from collections import deque, defaultdict
import random

# ============================ Configuración básica ============================
DEFAULT_EPS = "ε"     # configurable con --eps
OTHER = "OTHER"       # símbolo sintético para "otros" en DFA

# ============================ Tokenización / ER ===============================
@dataclass
class Token:
    kind: str
    value: Optional[str] = None
    # kind ∈ {"LIT","CLASS","ANY","ALT","CONCAT","STAR","PLUS","QMARK","LP","RP","REPEAT"}

def tokenize(expr: str) -> List[Token]:
    """
    Convierte la expresión regular cruda (infix) en una lista de tokens.
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
        elif c in " \t\r\n":
            i += 1
        elif c == ".":
            tokens.append(Token("ANY"))
            i += 1
        elif c in "|()*+?":
            kind_map = {
                "|": "ALT", "(": "LP", ")": "RP",
                "*": "STAR", "+": "PLUS", "?": "QMARK"
            }
            tokens.append(Token(kind_map[c]))
            i += 1
        elif c == "[":
            # clase de caracteres
            j = i + 1
            negate = False
            if j < n and expr[j] == "^":
                negate = True
                j += 1
            chars: List[str] = []
            while j < n and expr[j] != "]":
                if expr[j] == "\\" and j + 1 < n:
                    chars.append(expr[j + 1])
                    j += 2
                elif j + 2 < n and expr[j + 1] == "-" and expr[j + 2] != "]":
                    # rango a-z
                    start, end = expr[j], expr[j + 2]
                    chars.extend([chr(k) for k in range(ord(start), ord(end) + 1)])
                    j += 3
                else:
                    chars.append(expr[j])
                    j += 1
            if j >= n or expr[j] != "]":
                raise ValueError("Clase de caracteres no cerrada ']'")
            tokens.append(Token("CLASS", ("^" if negate else "") + "".join(chars)))
            i = j + 1
        elif c == "{":
            j = i + 1
            while j < n and expr[j] != "}":
                j += 1
            if j >= n:
                raise ValueError("Repetición '{...}' no cerrada '}'")
            content = expr[i:j + 1]  # incluye llaves
            tokens.append(Token("REPEAT", content))
            i = j + 1
        else:
            tokens.append(Token("LIT", c))
            i += 1
    return tokens

def render_atom(t: Token) -> str:
    if t.kind == "LIT":
        return t.value or ""
    if t.kind == "ANY":
        return "."
    if t.kind == "CLASS":
        return f"[{t.value}]"
    if t.kind == "REPEAT":
        return t.value or ""
    return t.kind

def needs_concat(prev: Optional[Token], cur: Token) -> bool:
    if prev is None:
        return False
    left = prev.kind in ("LIT", "CLASS", "ANY", "RP", "STAR", "PLUS", "QMARK", "REPEAT")
    right = cur.kind in ("LIT", "CLASS", "ANY", "LP")
    return left and right

def add_concat(tokens: List[Token]) -> List[Token]:
    """ Inserta tokens CONCAT explícitos donde corresponde. """
    out: List[Token] = []
    prev: Optional[Token] = None
    for t in tokens:
        if needs_concat(prev, t):
            out.append(Token("CONCAT"))
        out.append(t)
        prev = t
    return out

# =============================== Shunting Yard ===============================
PRECEDENCE = {
    "ALT": 1,      # |
    "CONCAT": 2,   # · (implícito)
    "STAR": 3,     # *
    "PLUS": 3,     # +
    "QMARK": 3,    # ?
    "REPEAT": 3,   # {m}, {m,}, {m,n}
}

@dataclass
class Step:
    out: List[str]
    ops: List[Token]

def shunting_yard(tokens: List[Token]) -> Tuple[List[str], List[Step]]:
    """
    Devuelve la lista de símbolos en postfix y un rastro de pasos.
    """
    steps: List[Step] = []
    out: List[str] = []
    ops: List[Token] = []

    def emit(tok: Token):
        if tok.kind == "LIT":
            out.append(tok.value or "")
        elif tok.kind == "CLASS":
            out.append(f"[{tok.value}]")
        elif tok.kind == "ANY":
            out.append(".")
        elif tok.kind in ("STAR", "PLUS", "QMARK", "REPEAT"):
            out.append(render_atom(tok))
        elif tok.kind in ("ALT", "CONCAT"):
            out.append(tok.kind)
        else:
            raise ValueError(f"Token no esperado en salida: {tok}")

    def prec(tok: Token) -> int:
        return PRECEDENCE.get(tok.kind, 0)

    for t in tokens:
        if t.kind in ("LIT", "CLASS", "ANY"):
            emit(t)
        elif t.kind in ("STAR", "PLUS", "QMARK", "REPEAT"):
            emit(t)
        elif t.kind == "LP":
            ops.append(t)
        elif t.kind == "RP":
            while ops and ops[-1].kind != "LP":
                emit(ops.pop())
            if not ops:
                raise ValueError("Paréntesis no balanceados: falta '('")
            ops.pop()  # saca '('
        elif t.kind in ("ALT", "CONCAT"):
            while ops and ops[-1].kind not in ("LP",) and prec(ops[-1]) >= prec(t):
                emit(ops.pop())
            ops.append(t)
        else:
            raise ValueError(f"Token desconocido: {t.kind}")

        steps.append(Step(out[:], ops[:]))

    while ops:
        if ops[-1].kind in ("LP", "RP"):
            raise ValueError("Paréntesis no balanceados al final")
        emit(ops.pop())
        steps.append(Step(out[:], ops[:]))

    return out, steps

# ===================== Expansión canónica de cuantificadores =================
def _repeat_expr(base: List[str], m: int, n: Optional[int]) -> List[str]:
    """
    Expande base^{m} (o {m,} o {m,n}) en postfix usando CONCAT y ALT con ε.
    """
    if m < 0 or (n is not None and n < m):
        raise ValueError("{m,n} inválido")

    out: List[str] = []
    # base^m
    for _ in range(m):
        out.extend(base)
        out.append("CONCAT")

    if n is None:
        # {m,} => base^m base*
        # Una sola STAR sobre un bloque "base"
        out.extend(base)
        out.append("STAR")
        out.append("CONCAT")
        return out

    if n == m:
        return out  # exactamente m

    # {m,n}  ==> base^m ( ε | base | base·base | ... | base^(n-m) )
    # OR encadenado: t0 | t1 | ... | tk
    # Cada ti = (base^i) con i ∈ [0, n-m]
    alts: List[List[str]] = []
    for i in range(0, n - m + 1):
        block: List[str] = []
        for _ in range(i):
            block.extend(base)
            block.append("CONCAT")
        alts.append(block if block else ["ε"])  # i=0 ⇒ ε

    # Alternar en postfix: a | b | c ⇒ a b ALT c ALT
    alt_seq: List[str] = []
    alt_seq.extend(alts[0])
    for part in alts[1:]:
        alt_seq.extend(part)
        alt_seq.append("ALT")

    out.extend(alt_seq)
    out.append("CONCAT")  # base^m CONCAT (alternativa)
    return out

def expand_postfix(postfix: List[str]) -> List[str]:
    """
    Reescribe postfix para eliminar PLUS, QMARK y REPEAT, dejando solo:
      - LIT, CLASS, ANY, ALT, CONCAT y STAR
    """
    out: List[str] = []
    stack: List[List[str]] = []
    i = 0
    n = len(postfix)

    def pop_atom() -> List[str]:
        if not stack:
            raise ValueError("Postfix inválido para expansión: falta operando")
        return stack.pop()

    while i < n:
        sym = postfix[i]
        if sym in ("ALT", "CONCAT"):
            b = pop_atom()
            a = pop_atom()
            stack.append(a + b + [sym])
        elif sym == "STAR":
            a = pop_atom()
            stack.append(a + ["STAR"])
        elif sym == "PLUS":
            a = pop_atom()
            stack.append(a + a + ["CONCAT", "STAR"])  # x x CONCAT STAR
        elif sym == "QMARK":
            a = pop_atom()
            stack.append(["ε"] + a + ["ALT"])
        elif sym.startswith("{") and sym.endswith("}"):
            # ej: {3}, {2,}, {1,4}
            a = pop_atom()
            content = sym[1:-1]
            if "," in content:
                parts = content.split(",")
                m = int(parts[0])
                if parts[1] == "":
                    nrep = None
                else:
                    nrep = int(parts[1])
            else:
                m = int(content)
                nrep = None
            stack.append(_repeat_expr(a, m, nrep))
        else:
            # átomo literal: símbolo, clase, '.', 'ε'
            stack.append([sym])
        i += 1

    if len(stack) != 1:
        raise ValueError("Postfix inválido tras expansión")
    return stack[0]

# =============================== Thompson AFN ================================
@dataclass
class State:
    id: int
    trans: Dict[str, List[int]]  # símbolo -> destinos
    eps: List[int]               # transiciones ε

class NFA:
    def __init__(self, eps_symbol: str = DEFAULT_EPS):
        self.states: List[State] = []
        self.start: int = self.add_state()
        self.accept: int = self.add_state()
        self.eps = eps_symbol

    def add_state(self) -> int:
        sid = len(self.states)
        self.states.append(State(sid, defaultdict(list), []))
        return sid

    def add_trans(self, u: int, sym: str, v: int):
        self.states[u].trans[sym].append(v)

    def add_eps(self, u: int, v: int):
        self.states[u].eps.append(v)

    @staticmethod
    def _concat(nfa1: "NFA", nfa2: "NFA") -> "NFA":
        # Conecta accept de nfa1 al start de nfa2 por ε
        out = NFA(nfa1.eps)
        out.states = [State(s.id, defaultdict(list, {k: v[:] for k, v in s.trans.items()}), s.eps[:]) for s in nfa1.states]
        offset = len(out.states)
        # redirige accept de nfa1 a start de nfa2
        out.add_eps(nfa1.accept, nfa2.start + offset)
        # copy nfa2
        for s in nfa2.states:
            new_trans = defaultdict(list, {k: [x + offset for x in v] for k, v in s.trans.items()})
            new_eps = [x + offset for x in s.eps]
            out.states.append(State(s.id + offset, new_trans, new_eps))
        out.start = nfa1.start
        out.accept = nfa2.accept + offset
        return out

    @staticmethod
    def _alt(nfa1: "NFA", nfa2: "NFA") -> "NFA":
        out = NFA(nfa1.eps)
        # copiar nfa1 y nfa2 con offsets
        off1 = 0
        for s in nfa1.states:
            out.states.append(State(s.id, defaultdict(list, {k: v[:] for k, v in s.trans.items()}), s.eps[:]))
        off2 = len(out.states)
        for s in nfa2.states:
            new_trans = defaultdict(list, {k: [x + off2 for x in v] for k, v in s.trans.items()})
            new_eps = [x + off2 for x in s.eps]
            out.states.append(State(s.id + off2, new_trans, new_eps))

        # nuevo start y accept
        new_start = len(out.states)
        out.states.append(State(new_start, defaultdict(list), []))
        new_accept = len(out.states)
        out.states.append(State(new_accept, defaultdict(list), []))

        # ε a starts originales
        out.add_eps(new_start, nfa1.start + off1)
        out.add_eps(new_start, nfa2.start + off2)
        # aceptar desde accepts originales
        out.add_eps(nfa1.accept + off1, new_accept)
        out.add_eps(nfa2.accept + off2, new_accept)

        out.start = new_start
        out.accept = new_accept
        return out

    @staticmethod
    def _star(nfa1: "NFA") -> "NFA":
        out = NFA(nfa1.eps)
        for s in nfa1.states:
            out.states.append(State(s.id, defaultdict(list, {k: v[:] for k, v in s.trans.items()}), s.eps[:]))
        # nuevo start y accept
        new_start = len(out.states)
        out.states.append(State(new_start, defaultdict(list), []))
        new_accept = len(out.states)
        out.states.append(State(new_accept, defaultdict(list), []))
        # ε-transiciones
        out.add_eps(new_start, nfa1.start)
        out.add_eps(new_start, new_accept)
        out.add_eps(nfa1.accept, nfa1.start)
        out.add_eps(nfa1.accept, new_accept)
        out.start = new_start
        out.accept = new_accept
        return out

    @staticmethod
    def _symbol(sym: str, eps: str) -> "NFA":
        out = NFA(eps)
        # reconstruir con dos estados (start->accept) por 'sym'
        out.states = []
        s = out.add_state()
        t = out.add_state()
        out.add_trans(s, sym, t)
        out.start, out.accept = s, t
        return out

    @staticmethod
    def _epsilon(eps: str) -> "NFA":
        out = NFA(eps)
        out.states = []
        s = out.add_state()
        t = out.add_state()
        out.add_eps(s, t)
        out.start, out.accept = s, t
        return out

    @staticmethod
    def build_from_postfix(postfix: List[str], eps_symbol: str = DEFAULT_EPS) -> "NFA":
        """
        Construye AFN por Thompson a partir de postfix
        """
        st: List[NFA] = []
        for sym in postfix:
            if sym == "ALT":
                b = st.pop()
                a = st.pop()
                st.append(NFA._alt(a, b))
            elif sym == "CONCAT":
                b = st.pop()
                a = st.pop()
                st.append(NFA._concat(a, b))
            elif sym == "STAR":
                a = st.pop()
                st.append(NFA._star(a))
            elif sym == "ε":
                st.append(NFA._epsilon(eps_symbol))
            else:
                st.append(NFA._symbol(sym, eps_symbol))
        if len(st) != 1:
            raise ValueError("Postfix inválida para Thompson")
        return st[0]

    # -------- Simulación AFN --------
    def eclosure(self, S: Set[int]) -> Set[int]:
        """ ε-clausura de un conjunto de estados. """
        stack = list(S)
        seen = set(S)
        while stack:
            u = stack.pop()
            for v in self.states[u].eps:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def move(self, S: Set[int], sym: str) -> Set[int]:
        T: Set[int] = set()
        for u in S:
            for v in self.states[u].trans.get(sym, []):
                T.add(v)
        return T

    def accepts(self, w: str, any_symbol: Optional[str] = None) -> bool:
        """
        Simula el AFN sobre w.
        """
        cur = self.eclosure({self.start})
        for ch in w:
            nxt = set()
            # movimientos explícitos
            nxt |= self.move(cur, ch)
            # si hay un comodín 'ANY'
            if any_symbol is not None:
                nxt |= self.move(cur, "ANY")
            cur = self.eclosure(nxt)
        return self.accept in cur

# ============================== Utilidades varias ============================
def expand_class(val: str) -> Set[str]:
    """
    Convierte el payload de un token CLASS en un conjunto de caracteres.
    """
    negate = False
    s = val
    if val.startswith("^"):
        negate = True
        s = val[1:]
    chars: Set[str] = set(s)
    return chars if not negate else set()  # Negado se resuelve más adelante

# ============================== Dibujo AFN ===================================
import matplotlib.pyplot as plt
import math

def layout_positions_nfa(nfa: NFA) -> Dict[int, Tuple[float, float]]:
    """
    Layout radial simple: start y accept más separados, intermedios en círculo.
    """
    n = len(nfa.states)
    if n <= 2:
        return {0: (0.0, 0.0), 1: (2.0, 0.0)}
    R = 4.0
    center = (0.0, 0.0)
    pos: Dict[int, Tuple[float, float]] = {}
    # coloca start y accept
    pos[nfa.start] = (-3.0, 0.0)
    pos[nfa.accept] = (3.0, 0.0)
    idx = 0
    for s in range(n):
        if s in (nfa.start, nfa.accept):
            continue
        angle = 2 * math.pi * (idx / max(1, n - 2))
        pos[s] = (center[0] + R * math.cos(angle), center[1] + R * math.sin(angle))
        idx += 1
    return pos

def draw_nfa_png(nfa: NFA, path: str):
    pos = layout_positions_nfa(nfa)
    fig, ax = plt.subplots(figsize=(8, 5))
    # dibujar estados
    for sid, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.35, fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f"S{sid}", ha="center", va="center")
    # doble círculo para accept
    ax.add_patch(plt.Circle(pos[nfa.accept], 0.28, fill=False, linewidth=2))

    # dibujar transiciones
    for s in nfa.states:
        x1, y1 = pos[s.id]
        for sym, dests in s.trans.items():
            for v in dests:
                x2, y2 = pos[v]
                ax.annotate("",
                            xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", lw=1.5))
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(xm, ym + 0.15, sym, fontsize=9, ha="center")
        for v in s.eps:
            x2, y2 = pos[v]
            ax.annotate("",
                        xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", lw=1, linestyle="dashed"))
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(xm, ym - 0.2, DEFAULT_EPS, fontsize=8, ha="center", style="italic")

    ax.set_aspect("equal")
    ax.axis("off")
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

# ============================== Front–End Parte A ============================
def regex_to_postfix(regex: str) -> Tuple[List[str], List[Step], List[str]]:
    """
    Pipeline A:
      regex (infix) -> tokens -> +CONCAT -> postfix -> expansión (sin +/?/{})
    Devuelve (postfix_original, steps_shunting, postfix_expandida)
    """
    toks = tokenize(regex)
    toks2 = add_concat(toks)
    postfix, steps = shunting_yard(toks2)
    expanded = expand_postfix(postfix)
    return postfix, steps, expanded

def build_nfa_from_regex(regex: str, eps_symbol: str = DEFAULT_EPS) -> Tuple[NFA, List[str], List[Step], List[str]]:
    postfix, steps, expanded = regex_to_postfix(regex)
    nfa = NFA.build_from_postfix(expanded, eps_symbol=eps_symbol)
    return nfa, postfix, steps, expanded

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

    def add_state(self, accept: bool=False) -> int:
        i = len(self.states)
        self.states[i] = DfaState(i, {}, accept)
        if self.start is None:
            self.start = i
        return i

    def add_trans(self, u: int, a: str, v: int):
        self.states[u].trans[a] = v

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
        for a in s.trans.keys():
            if a != "ANY":
                Σ.add(a)
    return Σ


def eclosure(nfa: NFA, S: Set[int]) -> Set[int]:
    return nfa.eclosure(S)


def move_set(nfa: NFA, S: Set[int], sym: str) -> Set[int]:
    T: Set[int] = set()
    for u in S:
        T |= set(nfa.states[u].trans.get(sym, []))
    return T


def build_dfa_from_nfa(nfa: NFA) -> Tuple[DFA, Dict[FrozenSet[int], int]]:
    Σ = nfa_alphabet(nfa)
    has_any = any("ANY" in s.trans for s in nfa.states)
    alphabet = set(Σ)
    if has_any:
        alphabet.add(OTHER)

    dfa = DFA(alphabet=alphabet)
    subset_id: Dict[FrozenSet[int], int] = {}

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

    # Partición inicial
    P: List[Set[int]] = []
    if F:  P.append(F)
    if NF: P.append(NF)
    W: List[Set[int]] = [min(F, key=len) if F else set()] if F else [NF]

    while W:
        A = W.pop()
        for a in Σ:
            # X = {q | δ(q,a) ∈ A}
            X = set()
            for q in states:
                if dfa.states[q].trans.get(a, None) in A:
                    X.add(q)
            newP: List[Set[int]] = []
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

    # Construcción del DFA mínimo
    block_id: Dict[int, int] = {}
    for i, block in enumerate(P):
        for s in block:
            block_id[s] = i

    min_dfa = DFA(alphabet=set(dfa.alphabet))
    new_state_map: Dict[int, int] = {}
    for i, block in enumerate(P):
        accept = any(dfa.states[s].accept for s in block)
        new_state_map[i] = min_dfa.add_state(accept)

    for i, block in enumerate(P):
        rep = next(iter(block))
        for a in dfa.alphabet:
            to_old = dfa.states[rep].trans[a]
            j = block_id[to_old]
            min_dfa.add_trans(new_state_map[i], a, new_state_map[j])

    if dfa.start is not None:
        min_dfa.start = new_state_map[block_id[dfa.start]]
    return min_dfa


# ============================== Dibujo AFD ===================================
def layout_positions_dfa(dfa: DFA) -> Dict[int, Tuple[float, float]]:
    adj = {i: set(s.trans.values()) for i, s in dfa.states.items()}
    dist: Dict[int, int] = {}
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
    levels: Dict[int, List[int]] = {}
    for s, d in dist.items():
        levels.setdefault(d, []).append(s)
    pos: Dict[int, Tuple[float, float]] = {}
    sep_x, sep_y = 2.6, 1.6
    for d in sorted(levels):
        ys = sorted(levels[d])
        n = len(ys)
        for i, s in enumerate(ys):
            y = (i - (n - 1) / 2.0) * sep_y
            x = d * sep_x
            pos[s] = (x, y)
    return pos


from matplotlib.patches import Circle, FancyArrowPatch, Arc

def draw_dfa_png(dfa: DFA, filename_png: str):
    pos = layout_positions_dfa(dfa)
    xs = [p[0] for p in pos.values()] + [0]
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
        a1 = d1c.states[u1].accept
        a2 = d2c.states[u2].accept
        if a1 != a2:
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
    out: List[str] = []
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