#!/bin/bash
# ============================================================
# TranscendPlexity AIMO3 — One-Click Kaggle Submission Setup
# ============================================================
#
# STEP 1: Get your Kaggle API key
#   1. Go to https://www.kaggle.com/settings
#   2. Scroll to "API" section
#   3. Click "Create New Token" — downloads kaggle.json
#
# STEP 2: Run this script with your credentials:
#   ./SUBMIT.sh YOUR_KAGGLE_USERNAME YOUR_KAGGLE_KEY
#
# STEP 3: The script will:
#   - Save credentials
#   - Join the competition
#   - Push notebook to Kaggle
#   - You then submit from Kaggle UI
# ============================================================

set -e

if [ $# -lt 2 ]; then
    echo "Usage: ./SUBMIT.sh <kaggle_username> <kaggle_api_key>"
    echo ""
    echo "Get your API key from: https://www.kaggle.com/settings → API → Create New Token"
    exit 1
fi

USERNAME="$1"
API_KEY="$2"

echo "🔑 Setting up Kaggle credentials for: $USERNAME"
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{"username": "$USERNAME", "key": "$API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json
echo "✅ Credentials saved to ~/.kaggle/kaggle.json"

# Activate venv
source /Users/evanpieser/venv/bin/activate

# Verify credentials
echo ""
echo "🔍 Verifying Kaggle access..."
kaggle competitions list --search "ai-mathematical-olympiad" 2>&1 | head -5
echo ""

# Join competition (may already be joined)
echo "📝 Accepting competition rules..."
kaggle competitions accept-rules -c ai-mathematical-olympiad-progress-prize-3 2>&1 || echo "(May need to accept rules manually on kaggle.com)"
echo ""

# Create kernel metadata
KERNEL_DIR="/Users/evanpieser/PROJECTS/aimo3/kaggle_submission"
mkdir -p "$KERNEL_DIR"

# Copy notebook
cp /Users/evanpieser/PROJECTS/aimo3/notebook.py "$KERNEL_DIR/notebook.py"

cat > "$KERNEL_DIR/kernel-metadata.json" << EOF
{
    "id": "${USERNAME}/transcendplexity-aimo3",
    "title": "TranscendPlexity AIMO3",
    "code_file": "notebook.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": false,
    "competition": "ai-mathematical-olympiad-progress-prize-3",
    "dataset_sources": [],
    "model_sources": ["openai/gpt-oss/transformers/120b/1"],
    "kernel_sources": [],
    "keywords": ["math", "olympiad", "reasoning", "transcendplexity"]
}
EOF

echo "📤 Pushing notebook to Kaggle..."
kaggle kernels push -p "$KERNEL_DIR" 2>&1
echo ""
echo "============================================================"
echo "✅ DONE! Your notebook is now on Kaggle."
echo ""
echo "Next steps:"
echo "  1. Go to: https://www.kaggle.com/code/${USERNAME}/transcendplexity-aimo3"
echo "  2. Click 'Edit' to verify the notebook looks correct"
echo "  3. Session Options → Accelerator → GPU H100"
echo "  4. Session Options → Internet → OFF"
echo "  5. Click 'Save & Run All'"
echo "  6. Once it runs, click 'Submit to Competition'"
echo "============================================================"
