#!/bin/bash
# aws/start.sh
# Start a previously stopped instance. The public IP usually changes on each
# start, so we re-query and update state/public_ip.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

if [ ! -f "$AWS_STATE_DIR/instance_id" ]; then
    echo "ERROR: no instance to start (state/instance_id missing)." >&2
    exit 1
fi
INSTANCE_ID=$(cat "$AWS_STATE_DIR/instance_id")
echo "Starting instance $INSTANCE_ID..."

aws ec2 start-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'StartingInstances[0].CurrentState.Name' --output text

aws ec2 wait instance-running \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$PUBLIC_IP" > "$AWS_STATE_DIR/public_ip"

echo "Instance running at $PUBLIC_IP"
echo "Connect with: ./aws/05_connect.sh"
