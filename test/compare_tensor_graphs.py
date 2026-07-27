#!/usr/bin/env python3
"""Compare pinned tinygrad and Polygrad canonical Tensor graph artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


# Classification never makes a finding pass. The canonical register owns the
# status and stage policy; open/pending/unregistered findings remain failures.
OP_PAIR_IDS = {
    ("REDUCE", "REDUCE_AXIS"): "PG-PARITY-006",
}


def load(path):
    return json.loads(Path(path).read_text())


def node_label(node):
    return node["op"], node["dtype"], node["arg"]


def classify(case, kind, path, tg_node, pg_node):
    pair = (
        tg_node["op"] if tg_node else None,
        pg_node["op"] if pg_node else None,
    )
    if pair in OP_PAIR_IDS:
        return OP_PAIR_IDS[pair]
    if (
        case == "rebuilt_after_realize"
        and kind == "sharing"
        and path == "root.src[1]"
        and tg_node
        and pg_node
        and tg_node["op"] == "ADD"
        and pg_node["op"] == "BUFFER"
    ):
        return "PG-PARITY-001"
    return None


def compare_graph(case, tg_graph, pg_graph):
    tg_nodes = tg_graph["nodes"]
    pg_nodes = pg_graph["nodes"]
    tg_to_pg, pg_to_tg = {}, {}
    findings = []
    visited_pairs = set()

    def add(kind, path, tg_id, pg_id, detail, finding_id=None):
        tg_node = tg_nodes[tg_id] if tg_id is not None else None
        pg_node = pg_nodes[pg_id] if pg_id is not None else None
        findings.append({
            "kind": kind,
            "path": path,
            "id": finding_id or classify(case, kind, path, tg_node, pg_node),
            "tinygrad": tg_node,
            "polygrad": pg_node,
            "detail": detail,
        })

    def visit(tg_id, pg_id, path):
        old_pg = tg_to_pg.get(tg_id)
        old_tg = pg_to_tg.get(pg_id)
        if old_pg is not None and old_pg != pg_id:
            add("sharing", path, tg_id, pg_id, f"tinygrad node already maps to Polygrad node {old_pg}")
            return
        if old_tg is not None and old_tg != tg_id:
            add("sharing", path, tg_id, pg_id, f"Polygrad node already maps to tinygrad node {old_tg}")
            return
        tg_to_pg[tg_id] = pg_id
        pg_to_tg[pg_id] = tg_id
        if (tg_id, pg_id) in visited_pairs:
            return
        visited_pairs.add((tg_id, pg_id))

        tg_node, pg_node = tg_nodes[tg_id], pg_nodes[pg_id]
        if node_label(tg_node) != node_label(pg_node):
            add("label", path, tg_id, pg_id, "op/dtype/arg differ")
        if len(tg_node["src"]) != len(pg_node["src"]):
            add(
                "arity", path, tg_id, pg_id,
                f"source count {len(tg_node['src'])} != {len(pg_node['src'])}",
            )
        common = min(len(tg_node["src"]), len(pg_node["src"]))
        for i in range(common):
            visit(tg_node["src"][i], pg_node["src"][i], f"{path}.src[{i}]")
        for i in range(common, len(tg_node["src"])):
            add(
                "missing_source", f"{path}.src[{i}]", tg_node["src"][i], None,
                "missing in Polygrad",
            )
        for i in range(common, len(pg_node["src"])):
            add(
                "extra_source", f"{path}.src[{i}]", None, pg_node["src"][i],
                "extra in Polygrad",
            )

    visit(tg_graph["root"], pg_graph["root"], "root")
    return findings


def validate_register(register, reference_root):
    if register.get("schema_version") != 1:
        raise RuntimeError("unsupported parity register schema")
    entries = register.get("entries", [])
    ids = [entry.get("id") for entry in entries]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise RuntimeError("parity register IDs are missing or duplicated")
    valid_statuses = {"approved", "open_debt", "pending_evidence"}
    if any(entry.get("status") not in valid_statuses for entry in entries):
        raise RuntimeError("parity register contains an unknown status")
    actual = subprocess.run(
        ["git", "-C", reference_root, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    expected = register["reference"]["commit"]
    if actual != expected:
        raise RuntimeError(f"stale tinygrad source lock: expected {expected}, got {actual}")
    return {entry["id"]: entry for entry in entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tinygrad")
    parser.add_argument("polygrad")
    parser.add_argument(
        "--register", default="test/fixtures/parity_divergences.json",
    )
    parser.add_argument("--reference-root", default="references/tinygrad_latest")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    tg, pg, register = load(args.tinygrad), load(args.polygrad), load(args.register)
    entries = validate_register(register, args.reference_root)
    if tg.get("schema_version") != 1 or pg.get("schema_version") != 1:
        raise RuntimeError("unsupported graph artifact schema")
    if tg["engine"] != "tinygrad" or pg["engine"] != "polygrad":
        raise RuntimeError("wrong engine artifacts")
    if set(tg["cases"]) != set(pg["cases"]):
        raise RuntimeError("case sets differ")

    report = {
        "schema_version": 1,
        "reference_commit": register["reference"]["commit"],
        "cases": {},
        "summary": {"pass": 0, "fail": 0, "allowed": 0, "unregistered": 0},
    }
    for case in sorted(tg["cases"]):
        tg_case, pg_case = tg["cases"][case], pg["cases"][case]
        if tg_case["stage"] != pg_case["stage"]:
            raise RuntimeError(f"{case}: stage differs")
        stage = tg_case["stage"]
        findings = compare_graph(
            case, tg_case["roots"]["physical"], pg_case["roots"]["physical"],
        )
        for finding in findings:
            finding_id = finding["id"]
            entry = entries.get(finding_id)
            if entry is None:
                finding["status"] = "unregistered"
                finding["allowed"] = False
                report["summary"]["unregistered"] += 1
            elif stage not in entry["stages"]:
                finding["status"] = "stage_mismatch"
                finding["allowed"] = False
                report["summary"]["unregistered"] += 1
            else:
                finding["status"] = entry["status"]
                finding["allowed"] = entry["status"] == "approved"
                if finding["allowed"]:
                    report["summary"]["allowed"] += 1
        passed = all(finding["allowed"] for finding in findings)
        report["cases"][case] = {
            "stage": stage,
            "passed": passed,
            "findings": findings,
        }
        report["summary"]["pass" if passed else "fail"] += 1
    encoded = json.dumps(report, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n")
        summary = report["summary"]
        print(
            f"graph parity: {summary['pass']} pass, {summary['fail']} fail, "
            f"{summary['allowed']} allowed, {summary['unregistered']} unregistered"
        )
        if summary["fail"]:
            for case, case_report in report["cases"].items():
                if case_report["passed"]:
                    continue
                finding = next(item for item in case_report["findings"] if not item["allowed"])
                tg_op = finding["tinygrad"]["op"] if finding["tinygrad"] else "<missing>"
                pg_op = finding["polygrad"]["op"] if finding["polygrad"] else "<missing>"
                print(
                    f"first failure: {case} [{case_report['stage']}] "
                    f"{finding['id'] or 'UNREGISTERED'} {finding['status']} "
                    f"{finding['path']}: {tg_op} != {pg_op}"
                )
                break
    else:
        print(encoded)
    return 0 if args.report_only or report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
