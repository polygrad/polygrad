#!/usr/bin/env python
"""Compare two benchmark JSON files and detect performance regressions.

Usage:
  python bench/bench_compare.py <baseline.json> <current.json> [--threshold 0.10]

Exit code 0: no regressions. Exit code 1: regression detected.
"""

import json
import sys


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compare_python(baseline, current, threshold):
    """Compare Python benchmark ratios. Returns (lines, n_regress)."""
    bp = baseline.get('python', {})
    cp = current.get('python', {})
    lines = []
    n_regress = 0
    n_improved = 0

    all_keys = sorted(set(list(bp.keys()) + list(cp.keys())))
    for key in all_keys:
        if key not in bp:
            lines.append(('NEW', key, '', '', ''))
            continue
        if key not in cp:
            lines.append(('GONE', key, '', '', ''))
            continue

        br = bp[key].get('ratio_vs_numpy')
        cr = cp[key].get('ratio_vs_numpy')

        if br is None or cr is None or br == 0:
            lines.append(('SKIP', key, _rfmt(br), _rfmt(cr), ''))
            continue

        change = (cr - br) / br
        if change > threshold:
            lines.append(('REGRESS', key, _rfmt(br), _rfmt(cr), f'+{change*100:.0f}%'))
            n_regress += 1
        elif change < -threshold:
            lines.append(('IMPROVED', key, _rfmt(br), _rfmt(cr), f'{change*100:.0f}%'))
            n_improved += 1
        else:
            lines.append(('PASS', key, _rfmt(br), _rfmt(cr), f'{change*100:+.0f}%'))

    return lines, n_regress, n_improved


def compare_js(baseline, current, threshold):
    """Compare JS WASM/native ratios. Returns (lines, n_regress)."""
    bj = baseline.get('js', {})
    cj = current.get('js', {})
    lines = []
    n_regress = 0
    n_improved = 0

    all_keys = sorted(set(list(bj.keys()) + list(cj.keys())))
    for key in all_keys:
        if key not in bj:
            lines.append(('NEW', key, '', '', ''))
            continue
        if key not in cj:
            lines.append(('GONE', key, '', '', ''))
            continue

        br = bj[key].get('wasm_native_ratio')
        cr = cj[key].get('wasm_native_ratio')

        if br is None or cr is None or br == 0:
            lines.append(('SKIP', key, _rfmt(br), _rfmt(cr), ''))
            continue

        change = (cr - br) / br
        if change > threshold:
            lines.append(('REGRESS', key, _rfmt(br), _rfmt(cr), f'+{change*100:.0f}%'))
            n_regress += 1
        elif change < -threshold:
            lines.append(('IMPROVED', key, _rfmt(br), _rfmt(cr), f'{change*100:.0f}%'))
            n_improved += 1
        else:
            lines.append(('PASS', key, _rfmt(br), _rfmt(cr), f'{change*100:+.0f}%'))

    return lines, n_regress, n_improved


def _rfmt(v):
    if v is None:
        return 'n/a'
    return f'{v:.2f}x'


def print_section(title, lines):
    if not lines:
        return
    print(f'\n  {title}')
    print(f'  {"status":<10} {"workload":<24} {"baseline":>10} {"current":>10} {"change":>8}')
    print('  ' + '-' * 64)
    for status, key, base, curr, change in lines:
        marker = '>>>' if status == 'REGRESS' else '   '
        print(f'{marker}{status:<9} {key:<24} {base:>10} {curr:>10} {change:>8}')


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <baseline.json> <current.json> [--threshold 0.10]')
        sys.exit(2)

    baseline_path = sys.argv[1]
    current_path = sys.argv[2]

    threshold = 0.10
    if '--threshold' in sys.argv:
        idx = sys.argv.index('--threshold')
        threshold = float(sys.argv[idx + 1])

    baseline = load_json(baseline_path)
    current = load_json(current_path)

    total_regress = 0
    total_improved = 0

    py_lines, py_reg, py_imp = compare_python(baseline, current, threshold)
    if py_lines:
        print_section(f'Python benchmarks (pg/numpy ratio, threshold={threshold*100:.0f}%)', py_lines)
        total_regress += py_reg
        total_improved += py_imp

    js_lines, js_reg, js_imp = compare_js(baseline, current, threshold)
    if js_lines:
        print_section(f'JS benchmarks (wasm/native ratio, threshold={threshold*100:.0f}%)', js_lines)
        total_regress += js_reg
        total_improved += js_imp

    print(f'\n  Summary: {total_regress} regression(s), {total_improved} improvement(s)')

    if total_regress > 0:
        print(f'  FAIL: {total_regress} regression(s) exceed {threshold*100:.0f}% threshold\n')
        sys.exit(1)
    else:
        print('  PASS: no regressions detected\n')
        sys.exit(0)


if __name__ == '__main__':
    main()
