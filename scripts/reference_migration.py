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
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HASH_SCHEME = "cpython-3.11-ast-dump-v1"


def require_hash_interpreter() -> None:
    # ast.dump is an interpreter-specific representation (3.12 adds
    # type_params). Reject drift instead of silently invalidating prior review.
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"{HASH_SCHEME} requires CPython 3.11; use "
            "references/.venv-tinygrad-py311/bin/python (Make: PARITY_PY)"
        )


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def commit(root: Path) -> str:
    return run("git", "rev-parse", "HEAD", cwd=root)


def python_symbols(path: Path) -> dict[str, str]:
    require_hash_interpreter()
    if not path.is_file():
        return {}
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    require_hash_interpreter()
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
            state = "added" if name not in old else "removed" if name not in new else "changed" if old[name] != new[name] else None
            if state:
                ret.append({
                    "path": rel, "symbol": name, "state": state,
                    "baseline_sha256": old.get(name), "target_sha256": new.get(name),
                    "routed": not lowered or any(token in name.lower() for token in lowered),
                })
    return ret


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_inventory(old_root: Path, new_root: Path, paths: list[str]) -> list[dict]:
    # Whole-file rows require full-diff review, including imports, module-level
    # control flow and matcher composition which a symbol census cannot certify.
    rows = []
    for rel in sorted(set(iter_wave_files(old_root, paths)) | set(iter_wave_files(new_root, paths))):
        old, new = file_hash(old_root / rel), file_hash(new_root / rel)
        if old != new:
            rows.append({"id": f"{rel}:@file", "baseline_sha256": old, "target_sha256": new})
    rows.extend({"id": f"{r['path']}:{r['symbol']}",
                 "baseline_sha256": r["baseline_sha256"], "target_sha256": r["target_sha256"]}
                for r in symbol_changes(old_root, new_root, paths, []))
    return rows


def source_manifest(root: Path) -> dict[str, str]:
    """Conservative execution-input boundary; excludes generated mirrors/artifacts.

    Each run records its built artifacts separately. Capture this manifest before
    execution and verify it again afterwards, never backfill it onto an old log.
    """
    paths = set()
    for directory in ("src", "py/polygrad", "js/src", "test", "py/tests", "js/test",
                      "scripts", "js/scripts", "references/tinygrad_latest/tinygrad"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.suffix in {".c", ".h", ".py", ".js", ".mjs", ".ts", ".sh"} and p.is_file())
    paths.update(root / p for p in (
        "Makefile", "CMakeLists.txt", "py/setup.py", "py/pyproject.toml", "js/package.json", "js/binding.gyp",
        "scripts/reference_migration_waves.json", "test/fixtures/parity_divergences.json",
    ) if (root / p).is_file())
    paths.update(p for p in (root / "js").glob("*") if p.suffix in {".c", ".h"} and p.is_file())
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(paths)}


def manifest_hash(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def graph_corpus(root: Path) -> dict[str, str]:
    # Share the runner's generated registrations. Isolated, site-free listing
    # imports no frontend and constructs no graphs; a second AST evaluator would
    # otherwise silently miss new registration forms.
    root = root.resolve()
    cases = json.loads(run(sys.executable, "-I", "-S", str(root / "test/tensor_graph_cases.py"), "--list", cwd=root))
    if (not isinstance(cases, dict) or not cases or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in cases.items())):
        raise ValueError("invalid graph case catalogue")
    return cases


def release_errors(ledger: dict, report: dict, evidence: dict, register: dict, root: Path = ROOT) -> list[str]:
    """Validate reviewed audit rows and source-bound execution records, not labels.

    A wave's source_audit is schema-1 JSON with baseline_commit, target_commit,
    and rows matching source_inventory exactly. Each row has a disposition and
    reason; equivalent rows name hashed counterpart sources and executed tests.
    Approved divergences must name an approved register ID and exact stage.

    Execution evidence is schema-1 JSON: source_inputs and checks keyed by the
    configured required_checks. Checks carry command, exit_code, passed/failed/
    skipped counts, passed_cases, source_sha256, hashed log and artifact records.
    physical_graph additionally binds report_sha256. Human review remains needed
    for semantic correspondence; hashes attest identity, not truth of a claim.
    """
    errors = list(ledger.get("reference_errors", []))
    if ledger.get("hash_scheme") != HASH_SCHEME:
        errors.append("ledger: missing or unsupported hash scheme")

    def check_file(record, label):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            errors.append(f"{label}: missing file evidence")
            return None
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"{label}: missing or outside-root file {record['path']}")
            return None
        if not record.get("sha256") or file_hash(path) != record["sha256"]:
            errors.append(f"{label}: stale file {record['path']}")
        return path

    def load_audit(name, label):
        if not isinstance(name, str) or not name:
            errors.append(f"{label}: missing source audit evidence")
            return {}
        path = (root / name).resolve()
        try:
            if not path.is_relative_to(root.resolve()):
                raise ValueError("outside repository")
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise ValueError("unsupported audit schema")
            return data
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: invalid source audit {name}: {exc}")
            return {}

    entries = {r["id"]: r for r in register.get("entries", [])}
    if len(entries) != len(register.get("entries", [])):
        errors.append("divergence register: duplicate IDs")
    if register.get("schema_version") != 1 or register.get("reference", {}).get("commit") != ledger["target"]["commit"]:
        errors.append("divergence register: stale reference or unsupported schema")

    def approved(row):
        entry = entries.get(row.get("divergence_id", row.get("id")), {})
        return entry.get("status") == "approved" and row.get("stage") in entry.get("stages", [])

    for field in ("waves", "required_checks", "rule_groups"):
        if not ledger.get(field):
            errors.append(f"{field}: empty required scope")
    for row in ledger.get("rule_groups", []) + ledger.get("matcher_aggregates", []):
        if row.get("status") != "pass":
            errors.append(f"{row['id']}: matcher validation failed")

    summary = {"pass": 0, "fail": 0, "allowed": 0, "unregistered": 0}
    cases = report.get("cases", {})
    try:
        if {name: case.get("stage") for name, case in cases.items()} != graph_corpus(root):
            errors.append("physical graph corpus: missing/extra cases or changed stages")
    except (OSError, ValueError, SyntaxError, AttributeError, subprocess.CalledProcessError) as exc:
        errors.append(f"physical graph corpus: cannot read catalogue: {exc}")
    if report.get("schema_version") != 1 or report.get("reference_commit") != ledger["target"]["commit"] or not cases:
        errors.append("physical graph: missing cases, stale reference or unsupported schema")
    for name, case in cases.items():
        passed = case.get("passed") is True
        summary["pass" if passed else "fail"] += 1
        for finding in case.get("findings", []):
            allowed = approved({**finding, "stage": case.get("stage")})
            summary["allowed" if allowed else "unregistered"] += 1
            if not allowed or finding.get("allowed") is not True or finding.get("status") != "approved":
                errors.append(f"physical graph {name}: unapproved or unregistered mismatch")
        if not passed:
            errors.append(f"physical graph {name}: failed")
    if report.get("summary") != summary:
        errors.append("physical graph: summary disagrees with cases/register")
    check_file(ledger.get("graph_report"), "physical graph report")

    inputs = source_manifest(root)
    if evidence.get("schema_version") != 1 or not inputs or evidence.get("source_inputs") != inputs:
        errors.append("execution evidence: missing/stale source inputs or unsupported schema")
    source_sha256 = manifest_hash(inputs)
    checks = evidence.get("checks", {})
    required = set(ledger.get("required_checks", []))
    for wave in ledger.get("waves", []):
        required.update(wave.get("required_checks", []))
    for name in sorted(required):
        result = checks.get(name)
        if not isinstance(result, dict):
            errors.append(f"{name}: missing required execution result")
            continue
        counts = [result.get(k) for k in ("passed", "failed", "skipped")]
        passed_cases = result.get("passed_cases", [])
        valid_cases = (isinstance(passed_cases, list) and all(isinstance(c, str) and c for c in passed_cases)
                       and len(set(passed_cases)) == len(passed_cases))
        if (not result.get("command") or type(result.get("exit_code")) is not int or result["exit_code"] != 0
                or any(type(n) is not int or n < 0 for n in counts) or counts[0] == 0 or counts[1] != 0
                or not valid_cases or len(passed_cases) != counts[0]):
            errors.append(f"{name}: failed, empty or inconsistent execution result")
        if not valid_cases:
            passed_cases = []
        if result.get("source_sha256") != source_sha256:
            errors.append(f"{name}: stale execution source generation")
        check_file(result.get("log"), name)
        if not result.get("artifacts"):
            errors.append(f"{name}: missing built artifacts")
        for artifact in result.get("artifacts", []):
            check_file(artifact, name)
        if name == "physical_graph" and (
                result.get("report_sha256") != ledger["graph_report"]["sha256"]
                or set(passed_cases) != {n for n, c in cases.items() if c.get("passed") is True}):
            errors.append("physical_graph: stale report binding or case set")

    for wave in ledger.get("waves", []):
        label = wave["id"]
        if wave.get("source_status") != "closed":
            errors.append(f"{label}: source audit is not closed")
        if wave.get("implementation_status") != "closed":
            errors.append(f"{label}: implementation is not closed")
        audit = load_audit(wave.get("source_audit"), label)
        if not audit:
            continue
        if audit.get("hash_scheme") != HASH_SCHEME:
            errors.append(f"{label}: missing or unsupported audit hash scheme")
        if (audit.get("baseline_commit") != ledger["baseline"]["commit"]
                or audit.get("target_commit") != ledger["target"]["commit"]):
            errors.append(f"{label}: stale audit reference")
        inventory = {r["id"]: r for r in wave["source_inventory"]}
        rows = audit.get("rows", [])
        ids = [r.get("id") for r in rows]
        if set(ids) != set(inventory) or len(ids) != len(set(ids)):
            errors.append(f"{label}: missing, extra or duplicate source dispositions")
        for row in rows:
            row_label = f"{label}/{row.get('id')}"
            expected = inventory.get(row.get("id"), {})
            if any(row.get(k) != expected.get(k) for k in ("baseline_sha256", "target_sha256")):
                errors.append(f"{row_label}: stale source disposition")
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                errors.append(f"{row_label}: missing review rationale")
            disposition = row.get("disposition")
            if disposition == "approved_divergence":
                if not approved(row):
                    errors.append(f"{row_label}: unapproved divergence")
            elif disposition == "equivalent":
                if not row.get("counterparts") or not row.get("tests"):
                    errors.append(f"{row_label}: missing counterpart/test evidence")
                for counterpart in row.get("counterparts", []):
                    path = check_file(counterpart, row_label)
                    symbol = counterpart.get("symbol")
                    if not symbol or (path and not re.search(rf"\b{re.escape(symbol)}\b", path.read_text())):
                        errors.append(f"{row_label}: missing counterpart symbol")
                for test in row.get("tests", []):
                    check_file(test, row_label)
                    check_name = test.get("check")
                    result = checks.get(check_name) or {}
                    if check_name not in required or test.get("case") not in (result.get("passed_cases") or []):
                        errors.append(f"{row_label}: acceptance test was not executed")
            elif disposition != "not_applicable":
                errors.append(f"{row_label}: unclassified source change")
    return errors


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
            f"changed symbols: {len(wave['changed_symbols'])} "
            f"({sum(r['routed'] for r in wave['changed_symbols'])} lexically routed); "
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
    release = ledger.get("release_check")
    if release:
        lines.extend(["", f"## Strict evidence check: {release['status']}", ""])
        lines.extend(f"- {error}" for error in release["errors"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/reference_migration_waves.json")
    parser.add_argument("--graph-report")
    parser.add_argument("--archbird-map", default="temp/archbird-polygrad.json")
    parser.add_argument("--json-output", default="temp/reference_migration/ledger.json")
    parser.add_argument("--markdown-output", default="temp/reference_migration/ledger.md")
    parser.add_argument("--strict", action="store_true", help="fail on incomplete source/runtime evidence")
    parser.add_argument("--evidence", default="temp/reference_migration/evidence.json")
    args = parser.parse_args()
    require_hash_interpreter()

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
        inventory = source_inventory(old_root, new_root, wave["tinygrad_paths"])
        # Routing is not an exclusion or an allowance. Keep shared dependencies
        # in the exhaustive inventory until their evidence satisfies the gate.
        for row in inventory:
            row["review_owner"] = wave.get("source_review_owners", {}).get(
                row["id"].rsplit(":", 1)[0], wave["id"]
            )
        waves.append({
            **wave,
            "diff_stat": diff_stat(new_root, old_commit, new_commit, wave["tinygrad_paths"]),
            "diff_ops": diff_ops(new_root, old_commit, new_commit, wave["tinygrad_paths"]),
            "upstream_commits": upstream_commits(
                new_root, old_commit, new_commit, wave["tinygrad_paths"]
            ),
            "changed_symbols": changes,
            "source_inventory": inventory,
            "source_hints": source_hints([r for r in changes if r["routed"]], wave["polygrad_owners"]),
            "archbird_impact": archbird_impact(archbird, wave["polygrad_owners"]),
            "graph_status": graph_status(report, wave["case_patterns"]),
        })

    rule_groups = [validate_rule_group(group, new_root) for group in config.get("rule_groups", [])]
    matcher_aggregates = [
        validate_matcher_aggregate(aggregate, rule_groups)
        for aggregate in config.get("matcher_aggregates", [])
    ]

    ledger = {
        "schema_version": 3,
        "hash_scheme": HASH_SCHEME,
        "required_checks": config.get("required_checks", []),
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
    evidence_path = ROOT / args.evidence
    evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
    register = json.loads((ROOT / "test/fixtures/parity_divergences.json").read_text())
    reference_errors = []
    for reference_root in {old_root, new_root, ROOT / register["reference"]["root"]}:
        if run("git", "status", "--porcelain", "--untracked-files=all", cwd=reference_root):
            reference_errors.append(f"{reference_root.relative_to(ROOT)}: reference checkout is dirty")
    if commit(ROOT / register["reference"]["root"]) != new_commit:
        reference_errors.append("graph reference checkout differs from migration target")
    ledger["reference_errors"] = reference_errors
    errors = release_errors(ledger, report, evidence, register, ROOT)
    ledger["release_check"] = {"status": "fail" if errors else "pass", "errors": errors}
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
    print(f"strict evidence: {ledger['release_check']['status']} ({len(errors)} findings)")
    if args.strict:
        for error in errors:
            print(f"  {error}")
    return 1 if (args.strict and errors) or any(
        row["status"] != "pass" for row in rule_groups + matcher_aggregates
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
