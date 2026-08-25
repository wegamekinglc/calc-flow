"""Common subexpression extraction over resolved row-local forests.

Structurally identical, non-trivial subtrees referenced at least twice are
materialized once as ``__cf_cse_N`` columns in tiered expression nodes so the
final fused node computes every shared subexpression exactly once. Tiers are
emitted deepest first; discovery order, naming, and passthrough order are all
deterministic functions of the declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

from calc_flow.symbolic.nodes import CStr, Node, build

_TRIVIAL = frozenset({"column_ref", "literal"})
_PREDICATE_KEY = ("predicate",)


@dataclass(frozen=True, slots=True)
class FusedSegment:
    """The extraction outcome: emission-ordered tiers, final selects, predicate."""

    tiers: tuple[tuple[tuple[str, Node], ...], ...]
    selects: tuple[tuple[str, Node], ...]
    predicate: Node | None


def expression_refs(tree: Node, /) -> frozenset[str]:
    """Collect every column reference name inside one resolved tree."""

    names: set[str] = set()

    def walk(node: Node) -> None:
        if node.op.name == "column_ref":
            value = node.attr("name")
            if isinstance(value, CStr):
                names.add(value.value)
            return
        for argument in node.args:
            walk(argument)

    walk(tree)
    return frozenset(names)


def _subtree_counts(forest: list[tuple[tuple[str, ...], Node]], /) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(node: Node) -> None:
        counts[node.digest] = counts.get(node.digest, 0) + 1
        for argument in node.args:
            walk(argument)

    for _, tree in forest:
        walk(tree)
    return counts


def _maximal_candidates(
    forest: list[tuple[tuple[str, ...], Node]],
    counts: dict[str, int],
    /,
) -> list[tuple[str, Node]]:
    """Shared non-trivial subtrees not contained in any shared subtree.

    A subtree that only occurs inside other shared subtrees is deferred: once
    the enclosing candidates are rewritten to references, the deferred subtree
    is rediscovered in a deeper (earlier-emitted) tier, so no materialized
    column ever aliases another.
    """

    contained: set[str] = set()

    def mark_descendants(node: Node) -> None:
        for argument in node.args:
            contained.add(argument.digest)
            mark_descendants(argument)

    chosen: list[tuple[str, Node]] = []
    seen: set[str] = set()

    def walk(node: Node) -> None:
        if counts.get(node.digest, 0) >= 2 and node.op.name not in _TRIVIAL:
            if node.digest not in seen:
                seen.add(node.digest)
                chosen.append((node.digest, node))
                mark_descendants(node)
            return
        for argument in node.args:
            walk(argument)

    for _, tree in forest:
        walk(tree)
    return [(digest, tree) for digest, tree in chosen if digest not in contained]


def _rewrite(tree: Node, replacements: dict[str, str], /) -> Node:
    replacement = replacements.get(tree.digest)
    if replacement is not None:
        return build("column_ref", (), {"name": CStr(replacement)})
    if not tree.args:
        return tree
    return build(
        tree.op.name,
        tuple(_rewrite(argument, replacements) for argument in tree.args),
        dict(tree.attrs.entries),
        version=tree.op.version,
    )


def extract_common(
    selects: tuple[tuple[str, Node], ...],
    predicate: Node | None,
    reserved: frozenset[str],
    /,
) -> FusedSegment:
    """Extract shared subexpressions into emission-ordered materialization tiers.

    ``reserved`` names (declared fields) are never used for materialized
    columns. Discovery-order tiers are reversed for emission so deeper shared
    subexpressions are computed before the tiers that reference them.
    """

    forest: list[tuple[tuple[str, ...], Node]] = [
        (("select", name), tree) for name, tree in selects
    ]
    if predicate is not None:
        forest.append((_PREDICATE_KEY, predicate))
    iterations: list[tuple[str, ...]] = []
    counter = 0

    def next_name() -> str:
        nonlocal counter
        while f"__cf_cse_{counter}" in reserved:
            counter += 1
        name = f"__cf_cse_{counter}"
        counter += 1
        return name

    while True:
        counts = _subtree_counts(forest)
        candidates = _maximal_candidates(forest, counts)
        if not candidates:
            break
        replacements: dict[str, str] = {}
        names: list[str] = []
        defs: list[tuple[tuple[str, ...], Node]] = []
        for digest, tree in candidates:
            name = next_name()
            replacements[digest] = name
            names.append(name)
            defs.append((("cse", name), tree))
        iterations.append(tuple(names))
        forest = [(key, _rewrite(tree, replacements)) for key, tree in forest]
        forest.extend(defs)

    by_key = {key: tree for key, tree in forest}
    tiers = tuple(
        tuple((name, by_key[("cse", name)]) for name in names)
        for names in reversed(iterations)
    )
    final_selects = tuple((name, by_key[("select", name)]) for name, _ in selects)
    final_predicate = by_key.get(_PREDICATE_KEY)
    return FusedSegment(tiers, final_selects, final_predicate)
