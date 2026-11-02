#!/usr/bin/env bash
# Fortnightly disk retention for retired models.
# Installs/loads a launchd agent that runs every 14 days (1,209,600 s).
#
#   bash scripts/install_purge_schedule.sh            # install + load
#   bash scripts/install_purge_schedule.sh --uninstall
#   bash scripts/install_purge_schedule.sh --run-now  # trigger immediately
#
# Override the retention window with GRACE_DAYS=30 before installing.
set -euo pipefail

cd "$(dirname "$0")/.."
PLATFORM_ROOT="$(pwd -P)"
LABEL="com.cartplatform.purge-retired"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PYTHON="${PYTHON:-${PLATFORM_ROOT}/../.venv/bin/python}"
GRACE_DAYS="${GRACE_DAYS:-14}"
INTERVAL=1209600   # 14 days
LOG_DIR="${PLATFORM_ROOT}/logs/retention"

case "${1:-}" in
  --uninstall)
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || \
      launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed ${LABEL}"
    exit 0
    ;;
  --run-now)
    launchctl kickstart -k "gui/$(id -u)/${LABEL}"
    echo "Triggered ${LABEL} — see ${LOG_DIR}/purge.log"
    exit 0
    ;;
esac

[ -x "$PYTHON" ] || { echo "Python not found at $PYTHON (set PYTHON=...)" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${PLATFORM_ROOT}/scripts/purge_retired_artifacts.py</string>
        <string>--apply</string>
        <string>--grace-days</string>
        <string>${GRACE_DAYS}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PLATFORM_ROOT}</string>
    <key>StartInterval</key>
    <integer>${INTERVAL}</integer>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd.err.log</string>
</dict>
</plist>
PLIST_EOF

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Installed ${LABEL}"
echo "  every ${INTERVAL}s (14 days), grace ${GRACE_DAYS} days"
echo "  plist : ${PLIST}"
echo "  log   : ${LOG_DIR}/purge.log"
