#!/usr/bin/env python3
"""Compare runtime Polygrad and pinned tinygrad op vocabularies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tomllib

def release_scope_errors(report, scope, version):
    """Exclude exact product capabilities without changing parity ownership/status."""
    errors = []
    if scope.get('schema_version') != 1 or scope.get('version') != version:
        errors.append('stale or invalid release scope version')
    if scope.get('reference_commit') != report.get('reference_commit'):
        errors.append('release scope reference differs from census')
    expected = {}
    for row in scope.get('excluded_vocabulary', []):
        key = (row.get('side'), row.get('op'), row.get('id'))
        if key in expected or not all(key) or not row.get('reason', '').strip():
            errors.append('invalid or duplicate vocabulary exclusion')
        expected[key] = row
    used = set()
    for row in report['findings']:
        key = (row['side'], row['op'], row['id'])
        if row['allowed'] and row['status'] == 'approved' and row['present']:
            continue
        if key not in expected or row['status'] != 'open_debt' or not row['present'] or key in used:
            errors.append(f'out-of-scope or changed vocabulary finding: {key}')
        used.add(key)
    if set(expected) != used:
        errors.append('release vocabulary exclusions no longer match the actual census')
    return errors


def main():
    from polygrad import _ffi
    from tinygrad.uop import Ops
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register", default="test/fixtures/parity_divergences.json",
    )
    parser.add_argument("--reference-root", default="references/tinygrad_latest")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--release-scope", help="exact candidate limits; does not approve parity allowances")
    args = parser.parse_args()

    register = json.loads(Path(args.register).read_text())
    entries = register["entries"]
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("parity register IDs are duplicated")
    valid_statuses = {"approved", "open_debt", "pending_evidence", "resolved"}
    if any(entry.get("status") not in valid_statuses for entry in entries):
        raise RuntimeError("parity register contains an unknown status")
    for entry in entries:
        if entry["status"] == "resolved" and not entry.get("resolution"):
            raise RuntimeError(f"{entry['id']}: resolved entry lacks closing evidence")
    actual_commit = subprocess.run(
        ["git", "-C", args.reference_root, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    expected_commit = register["reference"]["commit"]
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"stale tinygrad source lock: expected {expected_commit}, got {actual_commit}"
        )

    pg_ops = {
        name.decode("utf-8")
        for i in range(_ffi._lib.poly_op_count())
        if (name := _ffi._lib.poly_op_name(i))
    }
    tg_ops = {op.name for op in Ops}
    actual = {
        "tinygrad_only": sorted(tg_ops - pg_ops),
        "polygrad_only": sorted(pg_ops - tg_ops),
    }

    owners = {"tinygrad_only": {}, "polygrad_only": {}}
    for entry in entries:
        if entry["status"] == "resolved":
            continue
        for side, names in entry.get("vocabulary", {}).items():
            if side not in owners:
                raise RuntimeError(f"{entry['id']}: unknown vocabulary side {side}")
            for name in names:
                if name in owners[side]:
                    raise RuntimeError(
                        f"{side} op {name} owned by both {owners[side][name]['id']} and {entry['id']}"
                    )
                owners[side][name] = entry

    findings = []
    for side in ("tinygrad_only", "polygrad_only"):
        actual_names = set(actual[side])
        registered_names = set(owners[side])
        for name in sorted(actual_names | registered_names):
            entry = owners[side].get(name)
            present = name in actual_names
            if entry is None:
                status, finding_id, allowed = "unregistered", None, False
            elif not present:
                status, finding_id, allowed = "stale_register", entry["id"], False
            else:
                status, finding_id = entry["status"], entry["id"]
                allowed = entry["status"] == "approved"
            findings.append({
                "side": side,
                "op": name,
                "present": present,
                "id": finding_id,
                "status": status,
                "allowed": allowed,
            })

    summary = {
        "tinygrad": len(tg_ops),
        "polygrad": len(pg_ops),
        "shared": len(tg_ops & pg_ops),
        "tinygrad_only": len(actual["tinygrad_only"]),
        "polygrad_only": len(actual["polygrad_only"]),
        "unregistered": sum(item["status"] == "unregistered" for item in findings),
        "stale_register": sum(item["status"] == "stale_register" for item in findings),
        "blocking": sum(not item["allowed"] for item in findings),
    }
    report = {
        "schema_version": 1,
        "reference_commit": expected_commit,
        "actual": actual,
        "findings": findings,
        "summary": summary,
    }
    release_errors = None
    if args.release_scope:
        if args.report_only:
            parser.error('--release-scope cannot be combined with --report-only')
        scope = json.loads(Path(args.release_scope).read_text(encoding='utf-8'))
        root = Path(__file__).resolve().parents[1]
        version = tomllib.loads((root / 'py/pyproject.toml').read_text())['project']['version']
        release_errors = release_scope_errors(report, scope, version)
        if json.loads((root / 'js/package.json').read_text())['version'] != version:
            release_errors.append('Python and npm package versions disagree')
        report['release_scope'] = dict(status='failed' if release_errors else 'passed',
                                       errors=release_errors, excluded=scope['excluded_vocabulary'])
    encoded = json.dumps(report, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n")
        print(
            f"op census: {summary['shared']} shared, "
            f"{summary['tinygrad_only']} tinygrad-only, "
            f"{summary['polygrad_only']} Polygrad-only, "
            f"{summary['unregistered']} unregistered"
        )
        if summary["blocking"]:
            first = next(item for item in findings if not item["allowed"])
            print(
                f"first blocking op: {first['side']}:{first['op']} "
                f"{first['id'] or 'UNREGISTERED'} {first['status']}"
            )
    else:
        print(encoded)
    if release_errors is not None:
        print(f'release vocabulary: {report["release_scope"]["status"]}; parity debts remain open')
        return int(bool(release_errors))
    return 0 if args.report_only or summary["blocking"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
