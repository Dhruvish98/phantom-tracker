#!/bin/bash
# aws/03_setup_security_group.sh
#
# One-time: create a security group that allows SSH from your current public
# IP ONLY (port 22). Costs nothing. Re-run if your IP changes (residential ISPs
# often rotate IPs daily) - it'll add a new ingress rule for the new IP.
#
# We use the default VPC. If your account doesn't have a default VPC, create
# one in the AWS console first or modify this script to specify --vpc-id.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

# Get current public IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
if [[ ! "$MY_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: failed to detect public IP" >&2
    exit 1
fi
echo "Current public IP: $MY_IP"

# Find or create the security group
SG_ID=$(aws ec2 describe-security-groups \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating new security group '$SECURITY_GROUP_NAME'"
    SG_ID=$(aws ec2 create-security-group \
        --profile "$AWS_PROFILE" --region "$AWS_REGION" \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "SSH access for Phantom Tracker dev instance" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT_TAG}]" \
        --query GroupId --output text)
    echo "Created security group: $SG_ID"
else
    echo "Security group already exists: $SG_ID"
fi

# Save SG ID for the launcher to pick up
echo "$SG_ID" > "$AWS_STATE_DIR/security_group_id"

# Add ingress rule for current IP (no-op if already present)
if aws ec2 authorize-security-group-ingress \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32" 2>&1 | grep -qv "InvalidPermission.Duplicate"; then
    echo "Added SSH ingress rule for ${MY_IP}/32"
else
    echo "SSH ingress rule for ${MY_IP}/32 already exists"
fi

echo
echo "Security group ready: $SG_ID (allows SSH from $MY_IP)"
echo "If your IP changes later, re-run this script to add the new IP."
