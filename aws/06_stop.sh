#!/bin/bash
# aws/06_stop.sh
# Stop the instance gracefully. Stops compute billing immediately, but keeps
# the EBS volume (and your work) intact. Charges only ~$0.40/month for the
# 50GB EBS while stopped. Resume later with start.sh (continues where left off).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

if [ ! -f "$AWS_STATE_DIR/instance_id" ]; then
    echo "ERROR: no instance to stop (state/instance_id missing)." >&2
    exit 1
fi
INSTANCE_ID=$(cat "$AWS_STATE_DIR/instance_id")
echo "Stopping instance $INSTANCE_ID..."

aws ec2 stop-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'StoppingInstances[0].CurrentState.Name' --output text

echo "Stop initiated. Waiting for confirmation..."
aws ec2 wait instance-stopped \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID"
echo "Instance stopped. Compute billing has stopped; EBS billing continues at ~\$0.40/month."
