#!/bin/bash
# aws/04_launch_instance.sh
#
# Launch a g4dn.xlarge spot instance with the configured Deep Learning AMI.
# Saves the resulting instance ID and public IP to state/ for the helper
# scripts (connect.sh, stop.sh, terminate.sh) to consume.
#
# !!! THIS COSTS MONEY !!!
# Spot price for g4dn.xlarge in ap-south-1c is currently ~$0.25/hr.
# An idle instance still bills ~$0.25/hr until stopped or terminated.
#
# To stop billing: run ./stop.sh (preserves EBS volume + ~$0.40/month for storage)
#                  or  ./terminate.sh (deletes everything, no further charges)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

# Read security group ID from previous step
SG_ID_FILE="$AWS_STATE_DIR/security_group_id"
if [ ! -f "$SG_ID_FILE" ]; then
    echo "ERROR: security group not set up. Run aws/03_setup_security_group.sh first." >&2
    exit 1
fi
SG_ID=$(cat "$SG_ID_FILE")

echo "Launching $INSTANCE_TYPE in $AVAILABILITY_ZONE (region $AWS_REGION)"
echo "  AMI:           $AMI_ID"
echo "  Storage:       ${EBS_SIZE_GB}GB gp3"
echo "  Security grp:  $SG_ID"
echo "  Key pair:      $KEY_PAIR_NAME"
if [ "$USE_SPOT" == "true" ]; then
    echo "  Pricing:       SPOT (~\$0.25/hr; can be interrupted)"
else
    echo "  Pricing:       ON-DEMAND (\$0.526/hr; uninterruptible)"
fi
echo

# Build instance options - spot block. Use 'persistent' + 'stop' so AWS
# preserves the EBS volume if it reclaims the instance, and re-fulfills the
# request automatically when capacity returns. ('one-time' spot only supports
# 'terminate', which would wipe our data on interruption.)
SPOT_OPTS=""
if [ "$USE_SPOT" == "true" ]; then
    SPOT_OPTS='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}'
fi

# Block device for resized root volume
BLOCK_DEVICES="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"

INSTANCE_ID=$(aws ec2 run-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_PAIR_NAME" \
    --security-group-ids "$SG_ID" \
    --placement "AvailabilityZone=$AVAILABILITY_ZONE" \
    --block-device-mappings "$BLOCK_DEVICES" \
    $SPOT_OPTS \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=$INSTANCE_NAME}]" "ResourceType=volume,Tags=[{Key=Project,Value=$PROJECT_TAG}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "Instance launching: $INSTANCE_ID"
echo "$INSTANCE_ID" > "$AWS_STATE_DIR/instance_id"

echo "Waiting for instance to enter 'running' state (typically 30-60s)..."
aws ec2 wait instance-running \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$PUBLIC_IP" > "$AWS_STATE_DIR/public_ip"

echo
echo "============================================================="
echo "  Instance running"
echo "============================================================="
echo "  Instance ID:  $INSTANCE_ID"
echo "  Public IP:    $PUBLIC_IP"
echo "  SSH command:  ./aws/05_connect.sh"
echo "  Stop billing: ./aws/06_stop.sh   (keeps EBS, can resume later)"
echo "  Terminate:    ./aws/07_terminate.sh   (deletes everything)"
echo "============================================================="
echo
echo "Note: SSH may take another 30-60s to become responsive after instance"
echo "      reaches 'running' state (cloud-init still finalizing)."
