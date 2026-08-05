#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash monitor_tiara2_resources.sh [process_regex] [interval_seconds] [log_root]
# Example:
#   bash monitor_tiara2_resources.sh 'sample_fragments_by_bp.py' 2

PATTERN="${1:-sample_fragments_by_bp.py}"
INTERVAL="${2:-2}"
LOG_ROOT="${3:-/data/shouhanyu/Tiara2/log/resource_monitor}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${LOG_ROOT}/${STAMP}"
mkdir -p "$OUT"

if ! [[ "$INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[FATAL] interval must be numeric: $INTERVAL" >&2
  exit 2
fi

for cmd in ps pgrep awk free; do
  command -v "$cmd" >/dev/null || {
    echo "[FATAL] missing command: $cmd" >&2
    exit 2
  }
done

SSD_MAPPER="/dev/mapper/ssd_vg-ssd_lv"
SSD_KERNEL=""
if [[ -e "$SSD_MAPPER" ]]; then
  SSD_KERNEL="$(basename "$(readlink -f "$SSD_MAPPER")")"
fi

cat > "$OUT/run_info.txt" <<EOF
started_at=$(date --iso-8601=seconds)
monitor_pid=$$
pattern=$PATTERN
interval_seconds=$INTERVAL
hostname=$(uname -n 2>/dev/null || echo unknown)
ssd_mapper=$SSD_MAPPER
ssd_kernel=$SSD_KERNEL
ssd_mount=$(findmnt -no TARGET -T /ssd 2>/dev/null || true)
ssd_source=$(findmnt -no SOURCE -T /ssd 2>/dev/null || true)
EOF

printf '%s\n' \
  'timestamp,elapsed_s,pid_count,cpu_pct_sum,rss_gib,storage_read_mib_s,storage_write_mib_s,logical_read_mib_s,logical_write_mib_s,mem_available_gib,swap_used_gib' \
  > "$OUT/summary.csv"

printf '%s\n' \
  $'timestamp\tpid\tppid\tstate\tcpu_pct\tmem_pct\trss_kib\telapsed_s\tcomm\targs' \
  > "$OUT/processes.tsv"

BG_PIDS=()
if command -v iostat >/dev/null; then
  # -N renders LVM/device-mapper names, -y skips since-boot averages.
  stdbuf -oL iostat -N -y -dxm "$INTERVAL" > "$OUT/iostat.log" 2>&1 &
  BG_PIDS+=("$!")
else
  echo "iostat unavailable" > "$OUT/iostat.log"
fi

if command -v mpstat >/dev/null; then
  stdbuf -oL mpstat -P ALL "$INTERVAL" > "$OUT/mpstat.log" 2>&1 &
  BG_PIDS+=("$!")
else
  echo "mpstat unavailable" > "$OUT/mpstat.log"
fi

if command -v vmstat >/dev/null; then
  stdbuf -oL vmstat -w "$INTERVAL" > "$OUT/vmstat.log" 2>&1 &
  BG_PIDS+=("$!")
else
  echo "vmstat unavailable" > "$OUT/vmstat.log"
fi

START_EPOCH="$(date +%s)"
PREV_EPOCH="$(date +%s.%N)"
PREV_READ=0
PREV_WRITE=0
PREV_RCHAR=0
PREV_WCHAR=0

cleanup() {
  local code=$?
  for pid in "${BG_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  {
    echo "stopped_at=$(date --iso-8601=seconds)"
    echo "exit_code=$code"
  } >> "$OUT/run_info.txt"

  echo
  echo "[monitor] logs: $OUT"
  if [[ -s "$OUT/summary.csv" ]]; then
    awk -F, 'NR>1 {
      n++;
      cpu+=$4; if($4>maxcpu)maxcpu=$4;
      rss+=$5; if($5>maxrss)maxrss=$5;
      rd+=$6; wr+=$7;
    } END {
      if(n>0) printf("[summary] samples=%d avg_cpu=%.1f%% max_cpu=%.1f%% avg_rss=%.2fGiB max_rss=%.2fGiB avg_storage_read=%.1fMiB/s avg_storage_write=%.1fMiB/s\n", n,cpu/n,maxcpu,rss/n,maxrss,rd/n,wr/n);
    }' "$OUT/summary.csv"
  fi
}
trap cleanup EXIT INT TERM

echo "[monitor] pattern : $PATTERN"
echo "[monitor] interval: ${INTERVAL}s"
echo "[monitor] logs    : $OUT"
echo "[monitor] SSD     : ${SSD_KERNEL:-unknown} (${SSD_MAPPER})"
echo "[monitor] Ctrl+C to stop"

while true; do
  NOW_ISO="$(date --iso-8601=seconds)"
  NOW_EPOCH="$(date +%s.%N)"
  ELAPSED="$(( $(date +%s) - START_EPOCH ))"

  PIDS=()
  RAW_PIDS="$(pgrep -f -- "$PATTERN" 2>/dev/null || true)"
  while read -r p; do
    [[ -n "$p" && "$p" != "$$" && -r "/proc/$p/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *"monitor_tiara2_resources.sh"* ]] && continue
    PIDS+=("$p")
  done <<< "$RAW_PIDS"

  CPU_SUM="0"; RSS_KIB="0"
  READ_BYTES=0; WRITE_BYTES=0; RCHAR=0; WCHAR=0

  if ((${#PIDS[@]} > 0)); then
    PID_CSV="$(IFS=,; echo "${PIDS[*]}")"
    PS_ROWS="$(ps -ww -p "$PID_CSV" -o pid=,ppid=,stat=,pcpu=,pmem=,rss=,etimes=,comm=,args= 2>/dev/null || true)"
    if [[ -n "$PS_ROWS" ]]; then
      CPU_SUM="$(awk '{s+=$4} END{printf "%.2f",s+0}' <<< "$PS_ROWS")"
      RSS_KIB="$(awk '{s+=$6} END{printf "%.0f",s+0}' <<< "$PS_ROWS")"
      while IFS= read -r row; do
        [[ -n "$row" ]] || continue
        read -r pid ppid state pcpu pmem rss etimes comm args <<< "$row"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$NOW_ISO" "$pid" "$ppid" "$state" "$pcpu" "$pmem" "$rss" "$etimes" "$comm" "$args" \
          >> "$OUT/processes.tsv"
      done <<< "$PS_ROWS"
    fi

    for pid in "${PIDS[@]}"; do
      io="/proc/$pid/io"
      [[ -r "$io" ]] || continue
      rb="$(awk '$1=="read_bytes:"{print $2}' "$io" 2>/dev/null || echo 0)"
      wb="$(awk '$1=="write_bytes:"{print $2}' "$io" 2>/dev/null || echo 0)"
      rc="$(awk '$1=="rchar:"{print $2}' "$io" 2>/dev/null || echo 0)"
      wc="$(awk '$1=="wchar:"{print $2}' "$io" 2>/dev/null || echo 0)"
      READ_BYTES=$((READ_BYTES + ${rb:-0}))
      WRITE_BYTES=$((WRITE_BYTES + ${wb:-0}))
      RCHAR=$((RCHAR + ${rc:-0}))
      WCHAR=$((WCHAR + ${wc:-0}))
    done
  fi

  DT="$(awk -v n="$NOW_EPOCH" -v p="$PREV_EPOCH" 'BEGIN{d=n-p;if(d<=0)d=1;printf "%.6f",d}')"
  DR=$((READ_BYTES - PREV_READ)); ((DR<0)) && DR=0
  DW=$((WRITE_BYTES - PREV_WRITE)); ((DW<0)) && DW=0
  DRC=$((RCHAR - PREV_RCHAR)); ((DRC<0)) && DRC=0
  DWC=$((WCHAR - PREV_WCHAR)); ((DWC<0)) && DWC=0

  STORAGE_R="$(awk -v b="$DR"  -v d="$DT" 'BEGIN{printf "%.2f",b/1048576/d}')"
  STORAGE_W="$(awk -v b="$DW"  -v d="$DT" 'BEGIN{printf "%.2f",b/1048576/d}')"
  LOGICAL_R="$(awk -v b="$DRC" -v d="$DT" 'BEGIN{printf "%.2f",b/1048576/d}')"
  LOGICAL_W="$(awk -v b="$DWC" -v d="$DT" 'BEGIN{printf "%.2f",b/1048576/d}')"
  RSS_GIB="$(awk -v k="$RSS_KIB" 'BEGIN{printf "%.3f",k/1048576}')"
  MEM_AVAIL_GIB="$(awk '/MemAvailable:/{printf "%.3f",$2/1048576}' /proc/meminfo)"
  SWAP_TOTAL="$(awk '/SwapTotal:/{print $2}' /proc/meminfo)"
  SWAP_FREE="$(awk '/SwapFree:/{print $2}' /proc/meminfo)"
  SWAP_USED_GIB="$(awk -v t="$SWAP_TOTAL" -v f="$SWAP_FREE" 'BEGIN{printf "%.3f",(t-f)/1048576}')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$NOW_ISO" "$ELAPSED" "${#PIDS[@]}" "$CPU_SUM" "$RSS_GIB" \
    "$STORAGE_R" "$STORAGE_W" "$LOGICAL_R" "$LOGICAL_W" \
    "$MEM_AVAIL_GIB" "$SWAP_USED_GIB" >> "$OUT/summary.csv"

  if [[ -t 1 ]]; then clear; fi
  echo "Tiara2 resource monitor  $NOW_ISO"
  echo "pattern=$PATTERN  matched_pids=${#PIDS[@]}  interval=${INTERVAL}s"
  echo "CPU(sum)=${CPU_SUM}%  RSS=${RSS_GIB}GiB  MemAvailable=${MEM_AVAIL_GIB}GiB  SwapUsed=${SWAP_USED_GIB}GiB"
  echo "Process storage I/O: read=${STORAGE_R} MiB/s write=${STORAGE_W} MiB/s"
  echo "Process logical I/O: read=${LOGICAL_R} MiB/s write=${LOGICAL_W} MiB/s"
  echo
  if ((${#PIDS[@]} > 0)); then
    ps -ww -p "$(IFS=,; echo "${PIDS[*]}")" \
      -o pid,ppid,stat,psr,pcpu,pmem,rss,etime,comm,args 2>/dev/null || true
  else
    echo "No matching process. Start the target job or change the regex."
  fi
  echo
  echo "SSD/LVM latest samples:"
  tail -n 16 "$OUT/iostat.log" 2>/dev/null || true
  echo
  echo "Logs: $OUT"

  PREV_EPOCH="$NOW_EPOCH"
  PREV_READ="$READ_BYTES"; PREV_WRITE="$WRITE_BYTES"
  PREV_RCHAR="$RCHAR"; PREV_WCHAR="$WCHAR"
  sleep "$INTERVAL"
done
