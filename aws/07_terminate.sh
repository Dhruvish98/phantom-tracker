#!/bin/bash
# aws/07_terminate.sh
# DESTRUCTIVE: terminate the instance permanently. Deletes the EBS volume
# (anything saved on the instance is lost). All billing stops immediately.
# Use this when you're done with the project, or before re-launching with
# different settings.
#
# Confirms before acting because it's destructive.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

if [ ! -f "$AWS_STATE_DIR/instance_id" ]; then
    echo "ERROR: no instance to terminate (state/instance_id missing)." >&2
    exit 1
fi
INSTANCE_ID=$(cat "$AWS_STATE_DIR/instance_id")

echo "About to TERMINATE instance: $INSTANCE_ID"
echo "This will DELETE the EBS volume and all data on it."
echo
read -p "Type 'TERMINATE' (uppercase) to confirm: " confirm
if [ "$confirm" != "TERMINATE" ]; then
    echo "Cancelled."
    exit 1
fi

aws ec2 terminate-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text

# Clean up state files
rm -f "$AWS_STATE_DIR/instance_id" "$AWS_STATE_DIR/public_ip"
echo "Termination initiated; state files cleared. Goodbye, instance."
