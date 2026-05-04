#!/bin/bash
# aws/status.sh
# Quick read-only check: what's running, current state, costs so far this month.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

echo "=== Instances tagged Project=$PROJECT_TAG ==="
aws ec2 describe-instances \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --filters "Name=tag:Project,Values=$PROJECT_TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]' \
    --output table

echo
echo "=== Month-to-date estimated cost (USD) ==="
ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
START_OF_MONTH=$(date -u +%Y-%m-01)
TOMORROW=$(date -u -d "tomorrow" +%Y-%m-%d 2>/dev/null || date -u -v+1d +%Y-%m-%d)
aws ce get-cost-and-usage \
    --profile "$AWS_PROFILE" \
    --time-period "Start=${START_OF_MONTH},End=${TOMORROW}" \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
    --output text 2>/dev/null || echo "(Cost Explorer not yet enabled - takes 24h after first activity)"
