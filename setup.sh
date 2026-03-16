#!/bin/bash
# TranscendPlexity AIMO3 — Quick Setup
# Run this to configure your API keys for testing

echo "=========================================="
echo "  TranscendPlexity AIMO3 Setup"
echo "=========================================="
echo ""

ENVFILE="$(dirname "$0")/.env"

# NVIDIA NIM API
echo "📡 NVIDIA NIM API (for testing with DeepSeek-R1)"
echo "   Get your free API key at: https://build.nvidia.com"
echo "   1. Sign in with your NVIDIA dev account"
echo "   2. Find DeepSeek-R1 or any math model"
echo "   3. Click 'Get API Key' → copy the nvapi-... key"
echo ""
read -p "   NVIDIA NIM API Key (nvapi-...): " NVIDIA_KEY

if [ -n "$NVIDIA_KEY" ]; then
    cat > "$ENVFILE" << EOF
# TranscendPlexity AIMO3 Configuration

# NVIDIA NIM API (for local testing)
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=$NVIDIA_KEY
LLM_MODEL=deepseek-ai/deepseek-r1

# Kaggle (for submission)
# KAGGLE_USERNAME=
# KAGGLE_KEY=
EOF
    echo "   ✅ Saved to .env"
else
    echo "   ⏭️  Skipped (you can set this later in .env)"
fi

echo ""

# Kaggle
echo "📊 Kaggle API (for submitting to competition)"
echo "   Get your key at: https://www.kaggle.com/settings"
echo "   Under 'API' section → 'Create New Token'"
echo ""
read -p "   Kaggle Username: " KAGGLE_USER
read -p "   Kaggle API Key: " KAGGLE_KEY

if [ -n "$KAGGLE_USER" ] && [ -n "$KAGGLE_KEY" ]; then
    mkdir -p ~/.kaggle
    echo "{\"username\": \"$KAGGLE_USER\", \"key\": \"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "   ✅ Saved to ~/.kaggle/kaggle.json"

    # Also add to .env
    if [ -f "$ENVFILE" ]; then
        sed -i '' "s/# KAGGLE_USERNAME=.*/KAGGLE_USERNAME=$KAGGLE_USER/" "$ENVFILE"
        sed -i '' "s/# KAGGLE_KEY=.*/KAGGLE_KEY=$KAGGLE_KEY/" "$ENVFILE"
    fi
else
    echo "   ⏭️  Skipped"
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test sandbox:  python3 test_local.py --sandbox-only"
echo "  2. Test solver:   source .env && python3 test_local.py --problems 2"
echo "  3. Submit:        Upload notebook.py to Kaggle"
echo ""
