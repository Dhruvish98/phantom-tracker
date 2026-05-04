#!/bin/bash
# aws/05_connect.sh
# SSH into the launched instance using the saved private key + public IP.
# Pass any extra args to ssh, e.g. './05_connect.sh -L 8501:localhost:8501'
# for port forwarding (handy for the Streamlit eval dashboard).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if [ ! -f "$AWS_STATE_DIR/public_ip" ]; then
    echo "ERROR: no instance launched (state/public_ip missing). Run 04_launch_instance.sh first." >&2
    exit 1
fi
PUBLIC_IP=$(cat "$AWS_STATE_DIR/public_ip")
PRIVATE_KEY="$AWS_STATE_DIR/${KEY_PAIR_NAME}.pem"

# DL AMI Ubuntu 24.04 default user is 'ubuntu'
exec ssh -i "$PRIVATE_KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$PUBLIC_IP" "$@"
