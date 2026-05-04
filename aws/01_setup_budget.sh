#!/bin/bash
# aws/01_setup_budget.sh
#
# One-time: create a monthly cost budget with email alerts at 50% and 90%.
# This costs nothing - it's just an alarm that emails you if spend approaches
# the cap. Should be the FIRST thing you set up before any compute resources.
#
# Idempotent: safe to re-run (will report "already exists" if budget already set).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

verify_account || exit 1

ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
echo "Setting up \$$MONTHLY_BUDGET_USD/month budget for account $ACCOUNT_ID with email alerts to $BUDGET_NOTIFICATION_EMAIL"

# Check if budget already exists
if aws budgets describe-budget \
    --profile "$AWS_PROFILE" \
    --account-id "$ACCOUNT_ID" \
    --budget-name "$BUDGET_NAME" >/dev/null 2>&1; then
    echo "Budget '$BUDGET_NAME' already exists. Skipping create."
    exit 0
fi

# Create budget JSON inline
BUDGET_JSON=$(cat <<EOF
{
    "BudgetName": "$BUDGET_NAME",
    "BudgetLimit": {
        "Amount": "$MONTHLY_BUDGET_USD",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
}
EOF
)

# Notification thresholds: warn at 50%, scream at 90%, alarm at 100% (forecasted)
NOTIFICATIONS_JSON=$(cat <<EOF
[
    {
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 50,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$BUDGET_NOTIFICATION_EMAIL"}]
    },
    {
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 90,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$BUDGET_NOTIFICATION_EMAIL"}]
    },
    {
        "Notification": {
            "NotificationType": "FORECASTED",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 100,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$BUDGET_NOTIFICATION_EMAIL"}]
    }
]
EOF
)

aws budgets create-budget \
    --profile "$AWS_PROFILE" \
    --account-id "$ACCOUNT_ID" \
    --budget "$BUDGET_JSON" \
    --notifications-with-subscribers "$NOTIFICATIONS_JSON"

echo
echo "Budget created. You will receive emails at:"
echo "  - 50% (\$$(echo "$MONTHLY_BUDGET_USD * 0.5" | bc -l | sed 's/0*$//;s/\.$//'))"
echo "  - 90% (\$$(echo "$MONTHLY_BUDGET_USD * 0.9" | bc -l | sed 's/0*$//;s/\.$//'))"
echo "  - 100% forecasted"
echo
echo "Confirm any subscription emails AWS sends to $BUDGET_NOTIFICATION_EMAIL"
