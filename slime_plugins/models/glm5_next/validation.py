"""Opt-in raw rollout receipts for short native GLM5-next validation runs."""

from pathlib import Path

from slime.observability.rollout_data_utils import save_debug_rollout_data


def log_raw_rollouts(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    # Use the existing logger hook without enabling checkpoint-backed recovery.
    path = Path(args.save_debug_train_data).parent / "rollout-{rollout_id}.pt"
    save_debug_rollout_data(str(path), samples, rollout_id=rollout_id, evaluation=False, args=args)
    return False
