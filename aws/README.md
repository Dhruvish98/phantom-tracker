# Phantom Tracker - AWS workflow

Provisions a single GPU dev instance for testing, training, and demo recording.
All scripts use the SPJain AWS account (profile `shoppers-re`, account `622070274638`).

## One-time setup (run in order, ~5-10 min total)

```bash
cd aws/

# 1. Budget alert (no cost). REQUIRED FIRST.
./01_setup_budget.sh

# 2. SSH key pair (no cost). Generates state/phantom-tracker-key.pem locally.
./02_setup_keypair.sh

# 3. Security group allowing SSH from your current IP (no cost).
./03_setup_security_group.sh
```

After step 3, you have an isolated security group + a key pair registered with
AWS. No compute charges yet.

## Working with an instance

```bash
# Launch a spot g4dn.xlarge in Mumbai (~$0.25/hr while running)
./04_launch_instance.sh

# SSH in (use -L 8501:localhost:8501 to forward Streamlit port)
./05_connect.sh

# Stop billing when done for the day (preserves disk; ~$0.40/month while stopped)
./06_stop.sh

# Resume next time
./start.sh

# Permanently destroy (deletes disk; no further charges)
./07_terminate.sh

# Read-only status check
./status.sh
```

## Typical project budget

| Activity | Hours | Compute cost |
|---|---|---|
| Setup + workflow validation | 2 | ~$0.50 |
| Grounding DINO base/large eval runs | 6 | ~$1.50 |
| MOT17 ablation sweep | 10 | ~$2.50 |
| Demo recording + dry-run | 3 | ~$0.75 |
| Live demo session | 2 | ~$0.50 |
| EBS storage (50GB, ~30 days) | - | ~$5 |
| **Total estimated** | **~23 hr** | **~$10-12** |

The $30/month budget alert is a hard sanity check at 90% spend.

## Repo workflow on the instance

After SSH'ing in:

```bash
# Clone the repo
git clone https://github.com/Dhruvish98/phantom-tracker.git
cd phantom-tracker

# Install our extras on top of the DL AMI (PyTorch + CUDA already there)
pip install -r requirements.txt
pip install boxmot torchreid motmetrics streamlit transformers fpdf2 plotly

# Verify GPU works
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run a quick benchmark on one demo video
python main.py --input demos/slow_walkers.mp4 --classes person --no-display --output /tmp/test.mp4
```

## Cost-control non-negotiables

- **Always stop the instance when not actively using it** (`./06_stop.sh`). An
  idle running instance still bills compute time.
- **Watch the budget emails.** The 50% threshold should fire well before the
  end of the project; anything earlier means we're spending faster than planned.
- **Spot instance can be interrupted** by AWS when capacity tightens. If that
  happens during a long run, the instance enters `stopped` state automatically;
  resume with `./start.sh`. For long uninterruptible jobs (final demo
  recording), edit `config.sh` to set `USE_SPOT="false"` (~2x cost).

## State directory (`aws/state/`)

Local-only; tracks the launched instance ID, public IP, security group ID, and
holds the SSH private key. `.gitignore` excludes this directory; never commit it.

## Updating your IP (residential ISPs rotate them)

If you can't SSH after a day or two, your home IP probably changed:

```bash
./03_setup_security_group.sh   # adds your new IP to the SSH allowlist
```
