#!/bin/bash
# aws/02_setup_keypair.sh
#
# One-time: generate an SSH key pair locally and import the public key into
# AWS so we can SSH into spot instances later. The private key stays on this
# machine ONLY (never uploaded). Costs nothing.
#
# Idempotent: safe to re-run; will skip if both local key + AWS key exist.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

PRIVATE_KEY="$AWS_STATE_DIR/${KEY_PAIR_NAME}.pem"
PUBLIC_KEY="${PRIVATE_KEY}.pub"

# Generate local key if missing
if [ ! -f "$PRIVATE_KEY" ]; then
    echo "Generating new SSH key pair at $PRIVATE_KEY"
    ssh-keygen -t ed25519 -f "$PRIVATE_KEY" -N "" -C "phantom-tracker-aws"
    chmod 600 "$PRIVATE_KEY"
else
    echo "Local SSH key already exists at $PRIVATE_KEY"
fi

# Check if AWS already has a key registered with this name
if aws ec2 describe-key-pairs \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --key-names "$KEY_PAIR_NAME" >/dev/null 2>&1; then
    echo "AWS key pair '$KEY_PAIR_NAME' already registered. Skipping import."
    exit 0
fi

# Import public key into AWS. Note: on Windows + Git Bash, paths look like
# /d/foo/bar but the Windows-native AWS CLI needs D:\foo\bar. cygpath handles
# the translation. On Linux/macOS cygpath isn't present and we use the path as-is.
PUBLIC_KEY_PATH="$PUBLIC_KEY"
if command -v cygpath >/dev/null 2>&1; then
    PUBLIC_KEY_PATH=$(cygpath -w "$PUBLIC_KEY")
fi

echo "Importing public key into AWS as '$KEY_PAIR_NAME'"
aws ec2 import-key-pair \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --key-name "$KEY_PAIR_NAME" \
    --public-key-material "fileb://${PUBLIC_KEY_PATH}" \
    --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=$PROJECT_TAG}]" \
    --output table

echo
echo "Done. Private key: $PRIVATE_KEY (keep secret, never share or commit)"
echo "Public key registered with AWS as: $KEY_PAIR_NAME"
