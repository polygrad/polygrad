#!/usr/bin/env python3
"""Build a source-backed Tinygrad migration ledger.

Git/AST evidence identifies upstream symbol changes. The exact graph report
identifies runtime topology debt. The checked-in wave manifest only routes
those two evidence sources to Polygrad owners; it never declares parity.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def commit(root: Path) -> str:
    return run("git", "rev-parse", "HEAD", cwd=root)


def python_symbols(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    ret: dict[str, str] = {}

    def add(name: str, node: ast.AST) -> None:
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        ret[name] = hashlib.sha256(normalized.encode()).hexdigest()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node)
        elif isinstance(node, ast.ClassDef):
            add(node.name, node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(f"{node.name}.{child.name}", child)
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            add(f"{node.name}.{target.id}", child)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    add(target.id, node)
    return ret


def python_matcher_rules(
    path: Path,
    name: str,
    starred_expansions: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Return ordered literal PatternMatcher rows composed into name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def matcher_lists(node: ast.AST) -> list[ast.List | ast.Tuple]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return matcher_lists(node.left) + matcher_lists(node.right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PatternMatcher"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            return [node.args[0]]
        # Referenced matchers are validated as their own groups. This function
        # owns only rows declared literally at the selected assignment.
        if isinstance(node, ast.Name):
            return []
        raise RuntimeError(f"{path}:{name} has unsupported matcher composition: {ast.unparse(node)}")

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        matcher_nodes = matcher_lists(node.value)
        if not matcher_nodes:
            raise RuntimeError(f"{path}:{name} declares no literal PatternMatcher rows")
        ret = []
        entries = [entry for matcher_node in matcher_nodes for entry in matcher_node.elts]
        for source_index, entry in enumerate(entries):
            if isinstance(entry, ast.Starred):
                expansion = (starred_expansions or {}).get(str(source_index))
                if not expansion or not isinstance(entry.value, ast.GeneratorExp):
                    raise RuntimeError(
                        f"{path}:{name}[{source_index}] needs reviewed starred_expansions"
                    )
                generated = entry.value.elt
                if not isinstance(generated, (ast.List, ast.Tuple)) or not generated.elts:
                    raise RuntimeError(f"{path}:{name}[{source_index}] is not a matcher-row generator")
                pattern = generated.elts[0]
                for value in expansion:
                    normalized = ast.dump(pattern, annotate_fields=True, include_attributes=False)
                    normalized += f"\n{ast.unparse(entry.value.generators[0].iter)}={value}"
                    ret.append({
                        "index": len(ret),
                        "source_index": source_index,
                        "expansion": value,
                        "pattern": ast.unparse(pattern),
                        "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                    })
                continue
            if not isinstance(entry, (ast.List, ast.Tuple)) or not entry.elts:
                raise RuntimeError(f"{path}:{name}[{source_index}] is not a matcher row")
            pattern = entry.elts[0]
            normalized = ast.dump(pattern, annotate_fields=True, include_attributes=False)
            ret.append({
                "index": len(ret),
                "source_index": source_index,
                "pattern": ast.unparse(pattern),
                "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            })
        return ret
    raise RuntimeError(f"{path}:{name} not found")


def c_function_block(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    # Select a definition, never a call whose following text eventually reaches
    # another function body.
    match = re.search(
        rf"(?m)^[ \t]*(?:static[ \t]+)?[A-Za-z_]\w*(?:[ \t]+|[ \t]*\*[ \t]*)+"
        rf"{re.escape(name)}[ \t]*\([^;{{}}]*\)[ \t\r\n]*\{{",
        source,
    )
    if not match:
        raise RuntimeError(f"{path}:{name} not found")
    start = match.start()
    pos = source.find("{", match.start(), match.end())
    depth = 0
    state = "code"
    i = pos
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "code":
            if ch == "/" and nxt == "*": state, i = "block_comment", i + 2; continue
            if ch == "/" and nxt == "/": state, i = "line_comment", i + 2; continue
            if ch == '"': state, i = "string", i + 1; continue
            if ch == "'": state, i = "char", i + 1; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return source[start:i + 1]
        elif state == "block_comment" and ch == "*" and nxt == "/":
            state, i = "code", i + 2; continue
        elif state == "line_comment" and ch == "\n": state = "code"
        elif state in {"string", "char"}:
            if ch == "\\": i += 2; continue
            if (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                state = "code"
        i += 1
    raise RuntimeError(f"{path}:{name} has no closing brace")


def c_matcher_callbacks(block: str) -> list[str]:
    match = re.search(r"Poly(?:Named)?Rule\s+rules\s*\[\s*\]\s*=\s*\{(.*?)\};", block, re.S)
    if not match:
        raise RuntimeError("PolyRule rules[] not found in matcher function")
    body = match.group(1)
    entries: list[str] = []
    start = None
    depth = 0
    for index, char in enumerate(body):
        if char == "{":
            if depth == 0: start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                entries.append(body[start:index + 1])
                start = None
    callbacks = []
    for entry in entries:
        callback = re.search(r",\s*(\w+)\s*(?:\}|\))\s*$", entry, re.S)
        if not callback:
            raise RuntimeError(f"cannot parse C matcher row: {entry}")
        callbacks.append(callback.group(1))
    if callbacks:
        return callbacks

    # Named rows use POLY_RULE(pattern, callback); find the final top-level
    # argument without assuming the pattern itself has no nested calls.
    cursor = 0
    while (start := body.find("POLY_RULE(", cursor)) >= 0:
        pos = start + len("POLY_RULE(")
        depth = 1
        last_comma = -1
        while pos < len(body) and depth:
            if body[pos] == "(": depth += 1
            elif body[pos] == ")": depth -= 1
            elif body[pos] == "," and depth == 1: last_comma = pos
            pos += 1
        if depth or last_comma < 0:
            raise RuntimeError("cannot parse POLY_RULE matcher row")
        callback = body[last_comma + 1:pos - 1].strip()
        if not re.fullmatch(r"\w+", callback):
            raise RuntimeError(f"cannot parse POLY_RULE callback: {callback}")
        callbacks.append(callback)
        cursor = pos
    return callbacks


def validate_rule_group(group: dict, target_root: Path) -> dict:
    errors: list[str] = []
    tinygrad = group["tinygrad"]
    polygrad = group["polygrad"]
    tg_path = target_root / tinygrad["path"]
    pg_path = ROOT / polygrad["path"]
    try:
        rules = python_matcher_rules(
            tg_path,
            tinygrad["matcher"],
            tinygrad.get("starred_expansions"),
        )
    except RuntimeError as exc:
        return {**group, "status": "fail", "errors": [str(exc)]}
    expected_count = tinygrad.get("rule_count")
    if expected_count is not None and len(rules) != expected_count:
        errors.append(f"Tinygrad rule count {len(rules)} != {expected_count}")

    try:
        matcher_block = c_function_block(pg_path, polygrad["matcher"])
        callbacks = c_matcher_callbacks(matcher_block)
    except RuntimeError as exc:
        return {**group, "status": "fail", "errors": [str(exc)], "tinygrad_rules": rules}
    if callbacks != polygrad["callbacks"]:
        errors.append(f"Polygrad callbacks {callbacks} != {polygrad['callbacks']}")
    if len(callbacks) != len(rules):
        errors.append(f"one-to-one row count {len(callbacks)} != {len(rules)}")

    mapping = group.get("mapping")
    if mapping is None:
        mapping = [
            {"tinygrad_rule": index, "polygrad_symbols": [callback]}
            for index, callback in enumerate(callbacks)
        ]
    mapped = [row["tinygrad_rule"] for row in mapping]
    if mapped != list(range(len(rules))):
        errors.append(f"ordered Tinygrad rows {mapped} != {list(range(len(rules)))}")
    pg_source = pg_path.read_text(encoding="utf-8")
    for row in mapping:
        for symbol in row["polygrad_symbols"]:
            if not re.search(rf"\b{re.escape(symbol)}\s*\(", pg_source):
                errors.append(f"Polygrad counterpart {symbol} not found")
    for test in group.get("tests", []):
        if not (ROOT / test).exists(): errors.append(f"acceptance test {test} not found")

    return {
        **group,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "tinygrad_rules": rules,
        "tinygrad_matcher_sha256": hashlib.sha256(
            "\n".join(row["sha256"] for row in rules).encode()
        ).hexdigest(),
        "polygrad_callbacks": callbacks,
        "polygrad_matcher_sha256": hashlib.sha256(matcher_block.encode()).hexdigest(),
        "resolved_mapping": mapping,
    }


def validate_matcher_aggregate(aggregate: dict, groups: list[dict]) -> dict:
    by_id = {group["id"]: group for group in groups}
    errors: list[str] = []
    tinygrad_count = polygrad_count = 0
    for group_id in aggregate["components"]:
        group = by_id.get(group_id)
        if group is None:
            errors.append(f"missing component {group_id}")
            continue
        if group["status"] != "pass": errors.append(f"component {group_id} is not green")
        tinygrad_count += len(group.get("tinygrad_rules", []))
        polygrad_count += len(group.get("polygrad_callbacks", []))
    expected = aggregate["rule_count"]
    if tinygrad_count != expected: errors.append(f"Tinygrad aggregate {tinygrad_count} != {expected}")
    if polygrad_count != expected: errors.append(f"Polygrad aggregate {polygrad_count} != {expected}")
    return {
        **aggregate,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "tinygrad_rule_count": tinygrad_count,
        "polygrad_rule_count": polygrad_count,
    }


def iter_wave_files(root: Path, patterns: list[str]) -> list[str]:
    found: set[str] = set()
    for pattern in patterns:
        path = root / pattern
        if path.is_file():
            found.add(pattern)
        elif path.is_dir():
            found.update(str(p.relative_to(root)) for p in path.rglob("*.py"))
        else:
            found.update(
                str(p.relative_to(root))
                for p in root.rglob("*.py")
                if fnmatch.fnmatch(str(p.relative_to(root)), pattern)
            )
    return sorted(found)


def symbol_changes(old_root: Path, new_root: Path, paths: list[str], filters: list[str]) -> list[dict]:
    lowered = [x.lower() for x in filters]
    ret: list[dict] = []
    for rel in sorted(set(iter_wave_files(old_root, paths)) | set(iter_wave_files(new_root, paths))):
        old, new = python_symbols(old_root / rel), python_symbols(new_root / rel)
        for name in sorted(set(old) | set(new)):
            if lowered and not any(token in name.lower() for token in lowered):
                continue
            state = "added" if name not in old else "removed" if name not in new else "changed" if old[name] != new[name] else None
            if state:
                ret.append({"path": rel, "symbol": name, "state": state})
    return ret


def source_hints(changes: list[dict], owners: list[str]) -> list[dict]:
    owner_lines: dict[str, list[str]] = {}
    for owner in owners:
        path = ROOT / owner
        if path.is_file():
            owner_lines[owner] = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hints = []
    for change in changes:
        token = change["symbol"].split(".")[-1].lstrip("_")
        if len(token) < 3:
            continue
        variants = {token.lower(), token.lower().replace("_", "")}
        matches = []
        for owner, lines in owner_lines.items():
            for lineno, line in enumerate(lines, 1):
                flat = line.lower().replace("_", "")
                if any(v in line.lower() or v.replace("_", "") in flat for v in variants):
                    matches.append(f"{owner}:{lineno}")
                    if len(matches) == 5:
                        break
            if len(matches) == 5:
                break
        if matches:
            hints.append({**change, "polygrad_matches": matches})
    return hints


def case_matches(name: str, patterns: list[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def node_signature(node: dict | None) -> str:
    if not isinstance(node, dict):
        return "-"
    arg = str(node.get("arg", "None"))
    if len(arg) > 48:
        arg = arg[:45] + "..."
    return f"{node.get('op', '?')}[{node.get('dtype', '?')};{arg}]"


def finding_signature(finding: dict) -> str:
    return (
        f"{finding.get('kind', '?')}:"
        f"{node_signature(finding.get('tinygrad'))}->"
        f"{node_signature(finding.get('polygrad'))}"
    )


def graph_status(report: dict, patterns: list[str]) -> dict:
    selected = {name: case for name, case in report.get("cases", {}).items() if case_matches(name, patterns)}
    signatures = Counter()
    signature_cases: dict[str, set[str]] = {}
    for name, case in selected.items():
        for finding in case.get("findings", []):
            signature = finding_signature(finding)
            signatures[signature] += 1
            signature_cases.setdefault(signature, set()).add(name)
    return {
        "cases": len(selected),
        "pass": sum(bool(case.get("passed")) for case in selected.values()),
        "fail": sum(not bool(case.get("passed")) for case in selected.values()),
        "findings": sum(len(case.get("findings", [])) for case in selected.values()),
        "top_signatures": [
            {
                "signature": key,
                "count": value,
                "cases": sorted(signature_cases[key]),
            }
            for key, value in signatures.most_common(12)
        ],
        "failing_cases": [name for name, case in selected.items() if not case.get("passed")],
    }


def diff_ops(new_root: Path, old_commit: str, new_commit: str, paths: list[str]) -> dict:
    diff = run("git", "diff", "--unified=0", old_commit, new_commit, "--", *paths, cwd=new_root)
    added, removed = Counter(), Counter()
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        target = added if line[0] == "+" else removed
        target.update(re.findall(r"\bOps\.([A-Z][A-Z0-9_]*)\b", line[1:]))
    return {
        "added": [{"op": op, "count": count} for op, count in added.most_common()],
        "removed": [{"op": op, "count": count} for op, count in removed.most_common()],
    }


def archbird_impact(archbird: dict | None, owners: list[str]) -> dict:
    if not archbird:
        return {"production_neighbors": [], "routed_tests": []}
    owner_set = set(owners)
    neighbors = Counter()
    for edge in archbird.get("edges", []):
        source, target = edge.get("source"), edge.get("target")
        if source in owner_set and target and target not in owner_set:
            neighbors[target] += len(edge.get("sites", [])) or 1
        if target in owner_set and source and source not in owner_set:
            neighbors[source] += len(edge.get("sites", [])) or 1
    routed_tests = []
    for test in archbird.get("tests", []):
        routes = test.get("routes", {})
        count = sum(int(routes.get(owner, 0)) for owner in owners)
        if count:
            routed_tests.append({"path": test.get("path"), "witnesses": count})
    return {
        "production_neighbors": [
            {"path": path, "witnesses": count}
            for path, count in neighbors.most_common(30)
            if not path.startswith("test/") and "/test" not in path
        ],
        "routed_tests": sorted(routed_tests, key=lambda row: (-row["witnesses"], row["path"]))[:40],
    }


def diff_stat(new_root: Path, old_commit: str, new_commit: str, paths: list[str]) -> dict:
    output = run("git", "diff", "--numstat", old_commit, new_commit, "--", *paths, cwd=new_root)
    added = deleted = files = 0
    for line in output.splitlines():
        if not line:
            continue
        a, d, _ = line.split("\t", 2)
        files += 1
        added += int(a) if a.isdigit() else 0
        deleted += int(d) if d.isdigit() else 0
    return {"files": files, "added": added, "deleted": deleted}


def upstream_commits(new_root: Path, old_commit: str, new_commit: str, paths: list[str]) -> list[dict]:
    output = run(
        "git", "log", "--no-merges", "--format=%H%x09%cs%x09%s",
        f"{old_commit}..{new_commit}", "--", *paths, cwd=new_root,
    )
    ret = []
    for line in output.splitlines():
        if not line:
            continue
        commit_hash, date, subject = line.split("\t", 2)
        ret.append({"commit": commit_hash, "date": date, "subject": subject})
    return ret


def render_markdown(ledger: dict) -> str:
    lines = [
        "# Tinygrad Reference Migration Ledger",
        "",
        f"- Baseline: `{ledger['baseline']['path']}@{ledger['baseline']['commit'][:12]}`",
        f"- Target: `{ledger['target']['path']}@{ledger['target']['commit'][:12]}`",
        f"- Graph gate: `{ledger['graph_report']['path']}` — {ledger['graph_report']['summary']}",
        "",
        "## Ordered matcher correspondence",
        "",
    ]
    for group in ledger["rule_groups"]:
        lines.append(
            f"- `{group['id']}`: **{group['status']}** — "
            f"{len(group.get('tinygrad_rules', []))} Tinygrad rules / "
            f"{len(group.get('polygrad_callbacks', []))} C callbacks"
        )
        for error in group.get("errors", []): lines.append(f"  - {error}")
    for aggregate in ledger["matcher_aggregates"]:
        lines.append(
            f"- `{aggregate['id']}`: **{aggregate['status']}** — "
            f"{aggregate['tinygrad_rule_count']}/{aggregate['polygrad_rule_count']} "
            "Tinygrad/Polygrad rows"
        )
        for error in aggregate.get("errors", []): lines.append(f"  - {error}")
    lines.extend([
        "",
        "| Wave | Source audit | Behavior | Runtime | Upstream symbols | Graph cases | Exact | Findings | Primary mismatch |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for wave in ledger["waves"]:
        status = wave["graph_status"]
        primary = status["top_signatures"][0]["signature"] if status["top_signatures"] else "none"
        lines.append(
            f"| `{wave['id']}` | `{wave.get('source_status', 'open')}` | "
            f"`{wave.get('implementation_status', 'unclassified')}` | "
            f"`{wave.get('runtime_status', 'unclassified')}` | {len(wave['changed_symbols'])} | {status['cases']} | "
            f"{status['pass']} | {status['findings']} | `{primary}` |"
        )
    global_status = ledger["global_graph_status"]
    if global_status["top_signatures"]:
        lines.extend(["", "## Cross-case runtime mismatch motifs", ""])
        for row in global_status["top_signatures"]:
            cases = ", ".join(f"`{case}`" for case in row["cases"][:12])
            suffix = " …" if len(row["cases"]) > 12 else ""
            lines.append(f"- {row['count']} × `{row['signature']}` — {cases}{suffix}")
    for wave in ledger["waves"]:
        lines.extend(["", f"## {wave['id']}: {wave['title']}", ""])
        lines.append(f"Owners: {', '.join(f'`{x}`' for x in wave['polygrad_owners'])}")
        lines.append(
            f"Status: source audit `{wave.get('source_status', 'open')}`; "
            f"behavior `{wave.get('implementation_status', 'unclassified')}`; "
            f"runtime `{wave.get('runtime_status', 'unclassified')}`."
        )
        lines.append("")
        lines.append(
            f"Upstream diff: {wave['diff_stat']['files']} files, +{wave['diff_stat']['added']}/-{wave['diff_stat']['deleted']}; "
            f"changed routed symbols: {len(wave['changed_symbols'])}; "
            f"relevant commits: {len(wave['upstream_commits'])}."
        )
        status = wave["graph_status"]
        lines.append(
            f"Graph slice: {status['pass']}/{status['cases']} exact, {status['findings']} findings; "
            f"failing cases: {', '.join(status['failing_cases'][:20]) or 'none'}."
        )
        if status["top_signatures"]:
            lines.extend(["", "Top runtime motifs:"])
            for row in status["top_signatures"][:6]:
                cases = ", ".join(f"`{case}`" for case in row["cases"][:8])
                suffix = " …" if len(row["cases"]) > 8 else ""
                lines.append(f"- {row['count']} × `{row['signature']}` — {cases}{suffix}")
        if wave["source_hints"]:
            lines.extend(["", "Top automatic owner hints:"])
            for hint in wave["source_hints"][:20]:
                lines.append(
                    f"- `{hint['path']}:{hint['symbol']}` ({hint['state']}) → "
                    + ", ".join(f"`{match}`" for match in hint["polygrad_matches"])
                )
        if wave.get("acceptance_probes"):
            lines.extend(["", "Acceptance probes:"])
            for probe in wave["acceptance_probes"]:
                lines.append(f"- `{probe}`")
        if wave["upstream_commits"]:
            lines.extend(["", "Newest relevant upstream commits:"])
            for row in wave["upstream_commits"][:30]:
                lines.append(
                    f"- `{row['commit'][:12]}` ({row['date']}) {row['subject']}"
                )
        impact = wave["archbird_impact"]
        if impact["production_neighbors"]:
            lines.extend(["", "Archbird production impact candidates:"])
            for row in impact["production_neighbors"][:12]:
                lines.append(f"- `{row['path']}` ({row['witnesses']} static witnesses)")
        if impact["routed_tests"]:
            lines.extend(["", "Archbird routed test candidates:"])
            for row in impact["routed_tests"][:12]:
                lines.append(f"- `{row['path']}` ({row['witnesses']} static witnesses)")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/reference_migration_waves.json")
    parser.add_argument("--graph-report")
    parser.add_argument("--archbird-map", default="temp/archbird-polygrad.json")
    parser.add_argument("--json-output", default="temp/reference_migration/ledger.json")
    parser.add_argument("--markdown-output", default="temp/reference_migration/ledger.md")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_root, new_root = ROOT / config["baseline_ref"], ROOT / config["target_ref"]
    report_path = ROOT / (args.graph_report or config["graph_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    archbird_path = ROOT / args.archbird_map if args.archbird_map else None
    archbird = json.loads(archbird_path.read_text(encoding="utf-8")) if archbird_path and archbird_path.is_file() else None
    old_commit, new_commit = commit(old_root), commit(new_root)
    if report.get("reference_commit") != new_commit:
        raise RuntimeError(
            f"graph report is bound to {report.get('reference_commit')}, expected current target {new_commit}"
        )

    waves = []
    for wave in config["waves"]:
        changes = symbol_changes(
            old_root, new_root, wave["tinygrad_paths"], wave.get("symbol_patterns", [])
        )
        waves.append({
            **wave,
            "diff_stat": diff_stat(new_root, old_commit, new_commit, wave["tinygrad_paths"]),
            "diff_ops": diff_ops(new_root, old_commit, new_commit, wave["tinygrad_paths"]),
            "upstream_commits": upstream_commits(
                new_root, old_commit, new_commit, wave["tinygrad_paths"]
            ),
            "changed_symbols": changes,
            "source_hints": source_hints(changes, wave["polygrad_owners"]),
            "archbird_impact": archbird_impact(archbird, wave["polygrad_owners"]),
            "graph_status": graph_status(report, wave["case_patterns"]),
        })

    rule_groups = [validate_rule_group(group, new_root) for group in config.get("rule_groups", [])]
    matcher_aggregates = [
        validate_matcher_aggregate(aggregate, rule_groups)
        for aggregate in config.get("matcher_aggregates", [])
    ]

    ledger = {
        "schema_version": 2,
        "baseline": {"path": config["baseline_ref"], "commit": old_commit},
        "target": {"path": config["target_ref"], "commit": new_commit},
        "graph_report": {
            "path": str(report_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "summary": report.get("summary", {}),
        },
        "global_graph_status": graph_status(report, [""]),
        "archbird_map": None if not archbird_path or not archbird_path.is_file() else {
            "path": str(archbird_path.relative_to(ROOT)),
            "input_sha256": archbird.get("evidence", {}).get("input_sha256"),
            "config_sha256": archbird.get("evidence", {}).get("config_sha256"),
        },
        "rule_groups": rule_groups,
        "matcher_aggregates": matcher_aggregates,
        "waves": waves,
    }
    json_path, markdown_path = ROOT / args.json_output, ROOT / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(ledger), encoding="utf-8")
    print(f"wrote {json_path.relative_to(ROOT)} and {markdown_path.relative_to(ROOT)}")
    for wave in waves:
        status = wave["graph_status"]
        print(
            f"{wave['id']}: {status['pass']}/{status['cases']} exact, "
            f"{status['findings']} findings, {len(wave['changed_symbols'])} changed symbols"
        )
    for group in rule_groups:
        print(f"{group['id']}: {group['status']} ({len(group.get('tinygrad_rules', []))} rules)")
    for aggregate in matcher_aggregates:
        print(
            f"{aggregate['id']}: {aggregate['status']} "
            f"({aggregate['tinygrad_rule_count']}/{aggregate['polygrad_rule_count']} rows)"
        )
    return 1 if any(
        row["status"] != "pass" for row in rule_groups + matcher_aggregates
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
