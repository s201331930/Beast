#!/bin/bash
# ============================================================================
# TASI Daily Scanner - Runner Script
# ============================================================================
# Runs at 7:00 PM KSA time (after market close)
# Schedule: Sunday-Thursday (TASI trading days)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON_PATH=$(which python3)

# Create log directory if not exists
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
LOG_FILE="$LOG_DIR/scan_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee "$LOG_FILE"
echo "TASI Daily Scan Starting" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Change to script directory
cd "$SCRIPT_DIR"

# Run the scanner
$PYTHON_PATH "$SCRIPT_DIR/tasi_daily_scanner.py" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Scan Completed at $(date)" | tee -a "$LOG_FILE"
echo "Exit Code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "scan_*.log" -mtime +30 -delete 2>/dev/null

exit $EXIT_CODE
