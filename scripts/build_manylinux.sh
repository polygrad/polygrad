#!/usr/bin/env bash
# Called by make build-py-manylinux; publication is deliberately separate.
set -euo pipefail
shopt -s nullglob

versions=(39 310 311 312 313)

fail() { echo "$*" >&2; exit 1; }

if [[ ${1:-} == --inside ]]; then
  version=$2
  unset POLY_LIB PYTHONPATH PYTHONHOME
  export CC=gcc POLY_DEV=CPU DEV=CPU TMPDIR=/tmp PIP_NO_CACHE_DIR=1
  df -hT /tmp /var/tmp /work
  command -v gcc
  command -v auditwheel
  # Check the entire matrix before paying for the first compile.
  for v in "${versions[@]}"; do
    test -x "/opt/python/cp$v-cp$v/bin/python" || fail "Missing CPython $v in build image"
  done
  mkdir /work/wheels
  for v in "${versions[@]}"; do
    tag="cp$v-cp$v"
    python="/opt/python/$tag/bin/python"
    echo "Building and testing $tag"
    "$python" -m pip wheel --no-deps --no-cache-dir \
      --wheel-dir "/work/raw-$tag" "/release/polygrad-$version.tar.gz"
    raw=(/work/raw-"$tag"/*.whl)
    [[ ${#raw[@]} == 1 ]] || fail "Expected one raw wheel for $tag"
    auditwheel repair --plat manylinux_2_28_x86_64 \
      --wheel-dir /work/wheels "${raw[0]}"
    wheel=(/work/wheels/polygrad-"$version"-"$tag"-*.whl)
    [[ ${#wheel[@]} == 1 ]] || fail "Expected one repaired wheel for $tag"
    auditwheel show "${wheel[0]}" | tee "/work/audit-$tag.log"
    "$python" -m venv "/work/venv-$tag"
    "/work/venv-$tag/bin/python" -m pip install --only-binary=:all: "${wheel[0]}"
    "/work/venv-$tag/bin/python" -I /smoke.py
  done
  exit 0
fi

[[ -z ${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-} ]] ||
  fail 'Run make build-py-manylinux from the host, outside Apptainer.'
[[ $(uname -m) == x86_64 ]] || fail 'This target builds Linux x86_64 wheels only.'
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' py/pyproject.toml)
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'Missing/unsupported release version'
release=$(realpath -e "${1:-build/release/$version}")
[[ ! -e $release/wheels ]] || fail "$release/wheels already exists; preserve or move it before rebuilding."
test -f "$release/polygrad-$version.tar.gz"
sha256sum -c "$release/SHA256SUMS"
sdist_hash=$(sha256sum "$release/polygrad-$version.tar.gz")
grep -q "^${sdist_hash%% *} " "$release/SHA256SUMS" || fail 'Staged sdist has no recorded checksum'
command -v "$APPTAINER" >/dev/null || fail 'Apptainer is required on the host.'
mkdir -p "$MANYLINUX_WORK_DIR"
work_base=$(realpath "$MANYLINUX_WORK_DIR")
exec 9>"$work_base/.lock"
flock -n 9 || fail "Another wheel build owns $work_base"
run=$(mktemp -d "$work_base/run.XXXXXXXX")
image="$work_base/manylinux.sif"
echo "Wheel build evidence: $run"

build() {
  if [[ -f $image ]]; then
    [[ -f $image.ref && $(<"$image.ref") == "$MANYLINUX_IMAGE" ]] ||
      fail 'Cached image reference differs; choose another MANYLINUX_WORK_DIR.'
  else
    "$APPTAINER" pull "$run/image.sif" "$MANYLINUX_IMAGE"
    mv "$run/image.sif" "$image"
    printf '%s\n' "$MANYLINUX_IMAGE" > "$image.ref"
  fi
  # Containment otherwise puts /tmp, /var/tmp and HOME in a small session tmpfs.
  mkdir "$run/session"
  "$APPTAINER" exec --cleanenv --containall --workdir "$run/session" \
    --bind "$release:/release:ro" --bind "$run:/work" \
    --bind "$root/test/package_install_python.py:/smoke.py:ro" \
    --bind "$root/scripts/build_manylinux.sh:/recipe.sh:ro" \
    --pwd /work "$image" bash /recipe.sh --inside "$version"

  # A failed/incomplete build must never leave publishable wheels in release/.
  for v in "${versions[@]}"; do
    wheel=("$run/wheels/polygrad-$version-cp$v-cp$v-"*manylinux_2_28_x86_64*.whl)
    [[ ${#wheel[@]} == 1 ]] || fail "Missing or ambiguous tested wheel for CPython $v"
  done
  wheels=("$run/wheels/"*.whl)
  [[ ${#wheels[@]} == ${#versions[@]} ]] || fail 'Unexpected wheel count'
  (cd "$run/wheels" && sha256sum ./*.whl > SHA256SUMS)
  staged=$(mktemp -d "$release/.wheels.XXXXXXXX")
  cp "${wheels[@]}" "$run/wheels/SHA256SUMS" "$staged/"
  mv -T "$staged" "$release/wheels"
  echo "PASS: all five installed-wheel checks; artifacts: $release/wheels"
}

build 2>&1 | tee "$run/build.log"
