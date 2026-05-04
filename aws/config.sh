#!/bin/bash
# Shared configuration for all aws/* scripts. Source this from each script:
#     source "$(dirname "$0")/config.sh"
#
# Edit values here to change them globally. Most defaults are fine.

# === AWS account scoping ===
# IMPORTANT: this profile name is misleading - it actually points at the
# SPJain AWS account (account 622070274638). Other configured profiles on this
# machine belong to internships and must NEVER be used for Phantom Tracker.
export AWS_PROFILE="shoppers-re"
export AWS_REGION="ap-south-1"            # Mumbai - lowest latency from India

# === Resource naming ===
# All resources tagged with Project=PhantomTracker for cost tracking + cleanup.
export PROJECT_TAG="PhantomTracker"
export KEY_PAIR_NAME="phantom-tracker-key"
export SECURITY_GROUP_NAME="phantom-tracker-sg"
export INSTANCE_NAME="phantom-tracker-dev"
export BUDGET_NAME="phantom-tracker-monthly-budget"

# === Instance configuration ===
# g4dn.xlarge: 1x NVIDIA T4 (16GB VRAM), 4 vCPU, 16GB RAM
# Cheapest GPU instance with enough VRAM for our pipeline + Grounding DINO base
export INSTANCE_TYPE="g4dn.xlarge"
# AMI: Deep Learning AMI GPU PyTorch 2.10 (Ubuntu 24.04) - matches our local
# PyTorch version, has CUDA + nvidia drivers pre-installed.
# Verify the latest with: aws ec2 describe-images --owners amazon \
#   --filters "Name=name,Values=Deep Learning*PyTorch*Ubuntu*"
export AMI_ID="ami-0709a2015ae967b70"
export EBS_SIZE_GB="50"                   # ~30GB AMI base + 20GB headroom
export AVAILABILITY_ZONE="ap-south-1c"    # cheapest spot AZ in Mumbai (verified)
export USE_SPOT="true"                    # spot ~70% cheaper; can be interrupted

# === Budget guardrails ===
export MONTHLY_BUDGET_USD="30"
export BUDGET_NOTIFICATION_EMAIL="dhruvish.as24dxb014@spjain.org"

# === Local paths ===
# All script-managed state (instance ids, IPs, key files) lives here.
export AWS_STATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/state"
mkdir -p "$AWS_STATE_DIR"

# === Helper: confirm we're using the right account before any AWS call ===
verify_account() {
    local actual
    actual=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text 2>/dev/null)
    if [ "$actual" != "622070274638" ]; then
        echo "ERROR: AWS profile '$AWS_PROFILE' resolves to account '$actual', expected SPJain (622070274638)." >&2
        echo "Refusing to run anything on this account." >&2
        return 1
    fi
}
