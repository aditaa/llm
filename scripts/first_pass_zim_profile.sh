#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="data/raw_zim"
WARM_DIR="/mnt/ceph/llm/data/raw_zim"
REPORTS_DIR="artifacts/reports"
MOVE_EXCLUDED=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/first_pass_zim_profile.sh [--move-excluded] [--raw-dir DIR] [--warm-dir DIR] [--reports-dir DIR]

Description:
  Builds first-pass "talking-only" ZIM include/exclude manifests based on filename rules.
  Include target set:
    - gutenberg_en_all_YYYY-MM.zim
    - wikipedia_en_all_maxi_YYYY-MM.zim
    - wikipedia_en_simple_all_maxi_YYYY-MM.zim
    - wikiquote_en_all_maxi_YYYY-MM.zim
    - wikisource_en_all_maxi_YYYY-MM.zim
    - wiktionary_en_all_nopic_YYYY-MM.zim
    - vikidia_en_all_maxi_YYYY-MM.zim
    - wikivoyage_en_all_maxi_YYYY-MM.zim

Options:
  --move-excluded      Move excluded local ZIMs to warm storage with size verification.
  --raw-dir DIR        Local raw ZIM directory (default: data/raw_zim).
  --warm-dir DIR       Warm-storage raw ZIM directory (default: /mnt/ceph/llm/data/raw_zim).
  --reports-dir DIR    Output reports directory (default: artifacts/reports).
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --move-excluded)
      MOVE_EXCLUDED=1
      shift
      ;;
    --raw-dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --warm-dir)
      WARM_DIR="$2"
      shift 2
      ;;
    --reports-dir)
      REPORTS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$REPORTS_DIR"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "raw dir not found: $RAW_DIR" >&2
  exit 1
fi

inventory="$REPORTS_DIR/raw_zim_inventory.tsv"
include="$REPORTS_DIR/first_pass_include_zims.txt"
exclude="$REPORTS_DIR/first_pass_exclude_zims.txt"
include_targets="$REPORTS_DIR/first_pass_include_targets.txt"

find "$RAW_DIR" -maxdepth 1 -type f -name '*.zim' -printf '%f\t%s\n' | sort > "$inventory"

keep_re='^(gutenberg_en_all_[0-9]{4}-[0-9]{2}\.zim|wikipedia_en_all_maxi_[0-9]{4}-[0-9]{2}\.zim|wikipedia_en_simple_all_maxi_[0-9]{4}-[0-9]{2}\.zim|wikiquote_en_all_maxi_[0-9]{4}-[0-9]{2}\.zim|wikisource_en_all_maxi_[0-9]{4}-[0-9]{2}\.zim|wiktionary_en_all_nopic_[0-9]{4}-[0-9]{2}\.zim|vikidia_en_all_maxi_[0-9]{4}-[0-9]{2}\.zim|wikivoyage_en_all_maxi_[0-9]{4}-[0-9]{2}\.zim)$'

awk -F'\t' -v re="$keep_re" '$1 ~ re {print $1}' "$inventory" | sort > "$include"
awk -F'\t' -v re="$keep_re" '$1 !~ re {print $1}' "$inventory" | sort > "$exclude"

cat > "$include_targets" << 'TARGETS'
gutenberg_en_all_2025-11.zim
vikidia_en_all_maxi_2025-12.zim
wikipedia_en_all_maxi_2026-02.zim
wikipedia_en_simple_all_maxi_2026-02.zim
wikiquote_en_all_maxi_2026-01.zim
wikisource_en_all_maxi_2026-02.zim
wikivoyage_en_all_maxi_2025-12.zim
wiktionary_en_all_nopic_2026-02.zim
TARGETS

echo "include_count=$(wc -l < "$include")"
echo "exclude_count=$(wc -l < "$exclude")"

awk -F'\t' 'NR==FNR {k[$1]=1; next} {if(k[$1]) ki+=$2; else ex+=$2; all+=$2} END {printf "include_size_gib=%.2f\nexclude_size_gib=%.2f\ntotal_size_gib=%.2f\n", ki/1024/1024/1024, ex/1024/1024/1024, all/1024/1024/1024}' "$include" "$inventory"

if [[ "$MOVE_EXCLUDED" -eq 0 ]]; then
  echo "manifests written in $REPORTS_DIR"
  exit 0
fi

mkdir -p "$WARM_DIR"
log="$REPORTS_DIR/move_excluded_to_warm_$(date +%Y%m%d_%H%M%S).log"
active="$(
  ps -eo cmd \
    | rg -o "${RAW_DIR%/}/[^ ]+\\.zim" \
    | sed "s#${RAW_DIR%/}/##" \
    | sort -u \
    | tr '\n' ' ' \
    || true
)"

echo "[$(date -Iseconds)] start move excluded zims -> warm" | tee -a "$log"
echo "active_zims=${active:-none}" | tee -a "$log"

moved=0
removed_existing=0
skipped_active=0
failed=0
missing=0

while IFS= read -r fname; do
  [[ -z "$fname" ]] && continue

  src="$RAW_DIR/$fname"
  dst="$WARM_DIR/$fname"

  if [[ " ${active} " == *" ${fname} "* ]]; then
    echo "SKIP active $fname" | tee -a "$log"
    skipped_active=$((skipped_active + 1))
    continue
  fi

  if [[ ! -f "$src" ]]; then
    echo "MISS $fname" | tee -a "$log"
    missing=$((missing + 1))
    continue
  fi

  src_size=$(stat -c%s "$src")

  if [[ -f "$dst" ]]; then
    dst_size=$(stat -c%s "$dst" || echo -1)
    if [[ "$src_size" -eq "$dst_size" ]]; then
      rm -f "$src"
      echo "DROP local(existing same size) $fname" | tee -a "$log"
      removed_existing=$((removed_existing + 1))
      continue
    fi
  fi

  if rsync -ah --partial --inplace --info=stats1 "$src" "$WARM_DIR/" >> "$log" 2>&1; then
    dst_size=$(stat -c%s "$dst" || echo -1)
    if [[ "$src_size" -eq "$dst_size" ]]; then
      rm -f "$src"
      echo "MOVE ok $fname" | tee -a "$log"
      moved=$((moved + 1))
    else
      echo "FAIL size-mismatch $fname src=$src_size dst=$dst_size" | tee -a "$log"
      failed=$((failed + 1))
    fi
  else
    echo "FAIL rsync $fname" | tee -a "$log"
    failed=$((failed + 1))
  fi
done < "$exclude"

echo "[$(date -Iseconds)] done moved=$moved removed_existing=$removed_existing skipped_active=$skipped_active missing=$missing failed=$failed" | tee -a "$log"
df -h / "${WARM_DIR%/}" | tee -a "$log"
echo "log_file=$log" | tee -a "$log"
