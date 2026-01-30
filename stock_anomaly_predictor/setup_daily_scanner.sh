#!/bin/bash
# ============================================================================
# TASI Daily Scanner Setup Script
# ============================================================================
# This script sets up the daily scanner to run at 7:00 PM KSA (16:00 UTC)
# 
# Usage:
#   chmod +x setup_daily_scanner.sh
#   ./setup_daily_scanner.sh
#
# Requirements:
#   - Python 3.8+
#   - pip packages: yfinance, pandas, numpy, scipy, scikit-learn, pytz
#   - (Optional) SMTP credentials for email
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH=$(which python3)
SCANNER_SCRIPT="$SCRIPT_DIR/tasi_daily_scanner.py"
LOG_DIR="$SCRIPT_DIR/logs"
CONFIG_FILE="$SCRIPT_DIR/.env"

echo "=============================================="
echo "TASI Daily Scanner Setup"
echo "=============================================="
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$SCRIPT_DIR/output/daily_scans"

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.8+"
    exit 1
fi
echo "  Python: $($PYTHON_PATH --version)"

# Check dependencies
echo "Checking dependencies..."
$PYTHON_PATH -c "import yfinance, pandas, numpy, scipy, sklearn, pytz" 2>/dev/null || {
    echo "Installing missing dependencies..."
    pip3 install yfinance pandas numpy scipy scikit-learn pytz
}
echo "  All dependencies OK"

# Create environment file if not exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo "Creating environment configuration..."
    cat > "$CONFIG_FILE" << 'EOF'
# TASI Daily Scanner Configuration
# ================================
# Edit this file with your email credentials

# Email Settings (for Gmail, enable "Less secure apps" or use App Password)
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Recipient
RECIPIENT_EMAIL=n.aljudayi@gmail.com
EOF
    echo "  Created: $CONFIG_FILE"
    echo "  ⚠️  Please edit this file with your email credentials"
else
    echo "  Config file exists: $CONFIG_FILE"
fi

# Create runner script
RUNNER_SCRIPT="$SCRIPT_DIR/run_daily_scan.sh"
cat > "$RUNNER_SCRIPT" << EOF
#!/bin/bash
# Auto-generated runner script for TASI Daily Scanner
# Runs at 7:00 PM KSA time (after market close)

cd "$SCRIPT_DIR"

# Load environment variables
if [ -f "$CONFIG_FILE" ]; then
    export \$(cat "$CONFIG_FILE" | grep -v '^#' | xargs)
fi

# Run scanner with logging
LOG_FILE="$LOG_DIR/scan_\$(date +%Y%m%d_%H%M%S).log"

echo "Starting TASI Daily Scan at \$(date)" | tee "\$LOG_FILE"
$PYTHON_PATH "$SCANNER_SCRIPT" 2>&1 | tee -a "\$LOG_FILE"
echo "Completed at \$(date)" | tee -a "\$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "scan_*.log" -mtime +30 -delete 2>/dev/null || true
EOF
chmod +x "$RUNNER_SCRIPT"
echo "  Created runner: $RUNNER_SCRIPT"

# Setup cron job
echo ""
echo "Setting up cron job..."
echo "  Schedule: 7:00 PM KSA (16:00 UTC) Sunday-Thursday"

# Create cron entry (16:00 UTC = 19:00 KSA)
CRON_ENTRY="0 16 * * 0-4 $RUNNER_SCRIPT >> $LOG_DIR/cron.log 2>&1"

# Check if cron entry already exists
(crontab -l 2>/dev/null | grep -v "$SCANNER_SCRIPT") | { cat; echo "$CRON_ENTRY"; } | crontab -

echo "  ✓ Cron job installed"

# Verify
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Scanner:  $SCANNER_SCRIPT"
echo "  Runner:   $RUNNER_SCRIPT"
echo "  Logs:     $LOG_DIR"
echo "  Schedule: Daily at 19:00 KSA (Sun-Thu)"
echo ""
echo "Current cron jobs:"
crontab -l | grep -E "(tasi|TASI)" || echo "  (none matching 'tasi')"
echo ""
echo "To test manually:"
echo "  $RUNNER_SCRIPT"
echo ""
echo "To configure email:"
echo "  nano $CONFIG_FILE"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/scan_*.log"
