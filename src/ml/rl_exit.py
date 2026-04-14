"""
Reinforcement Learning Exit Agent — Q-learning for position exit decisions.

Decides HOLD / EXIT / TIGHTEN for open F&O positions using a tabular
Q-learning agent trained on 8 state features derived from live position data.

State space (8 features):
  1. unrealized_pnl_pct — current P&L as % of entry price
  2. bars_held_norm — how long the position has been open (normalized 0-1)
  3. premium_momentum — rate of change of option premium
  4. premium_volatility — recent volatility of the premium
  5. distance_to_sl — how close price is to stop-loss (0=at SL, 1=far)
  6. distance_to_tgt — how close price is to target (0=at TGT, 1=far)
  7. trailing_active — whether trailing stop is engaged (0 or 1)
  8. peak_gain_pct — max unrealized gain seen during this trade

Action space:
  0 = HOLD  — keep position open
  1 = EXIT  — close position immediately
  2 = TIGHTEN — tighten stop-loss closer to current price

Reward design:
  EXIT    → realized P&L %
  HOLD    → -0.001 per bar (time penalty encourages decisive exits)
  TIGHTEN → +0.002 if currently profitable (reward risk reduction)
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("josho.ml.rl_exit")

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Actions
HOLD = 0
EXIT = 1
TIGHTEN = 2
ACTION_NAMES = {HOLD: "HOLD", EXIT: "EXIT", TIGHTEN: "TIGHTEN"}

# State feature count
NUM_FEATURES = 8

# Discretization bins per feature
NUM_BINS = 10

# Feature index names (for logging/debugging)
FEATURE_NAMES = [
    "unrealized_pnl_pct",
    "bars_held_norm",
    "premium_momentum",
    "premium_volatility",
    "distance_to_sl",
    "distance_to_tgt",
    "trailing_active",
    "peak_gain_pct",
]

# Feature ranges for discretization [min, max]
# These define the expected range for each feature; values outside are clipped.
FEATURE_RANGES = [
    (-0.20, 0.20),   # unrealized_pnl_pct: -20% to +20%
    (0.0, 1.0),      # bars_held_norm: 0 to 1
    (-0.05, 0.05),   # premium_momentum: -5% to +5% per bar
    (0.0, 0.10),     # premium_volatility: 0 to 10%
    (0.0, 1.0),      # distance_to_sl: 0 (at SL) to 1 (far)
    (0.0, 1.0),      # distance_to_tgt: 0 (at TGT) to 1 (far)
    (0.0, 1.0),      # trailing_active: binary
    (0.0, 0.30),     # peak_gain_pct: 0 to 30%
]


@dataclass(frozen=True)
class PositionState:
    """Immutable snapshot of a position's state for the RL agent."""
    unrealized_pnl_pct: float
    bars_held_norm: float
    premium_momentum: float
    premium_volatility: float
    distance_to_sl: float
    distance_to_tgt: float
    trailing_active: float
    peak_gain_pct: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for discretization."""
        return np.array([
            self.unrealized_pnl_pct,
            self.bars_held_norm,
            self.premium_momentum,
            self.premium_volatility,
            self.distance_to_sl,
            self.distance_to_tgt,
            self.trailing_active,
            self.peak_gain_pct,
        ], dtype=np.float64)


def _discretize(state_array: np.ndarray) -> tuple:
    """
    Convert continuous state features into discrete bin indices.
    Each feature is clipped to its expected range, then mapped to one of
    NUM_BINS bins. The result is a tuple used as a Q-table key.
    """
    bins = []
    for i in range(NUM_FEATURES):
        low, high = FEATURE_RANGES[i]
        val = float(np.clip(state_array[i], low, high))

        if high == low:
            bin_idx = 0
        else:
            normalized = (val - low) / (high - low)
            bin_idx = int(normalized * (NUM_BINS - 1))
            bin_idx = min(bin_idx, NUM_BINS - 1)

        bins.append(bin_idx)

    return tuple(bins)


def compute_reward(action: int, realized_pnl_pct: float, is_profitable: bool) -> float:
    """
    Compute the reward for a given action.

    Args:
        action: HOLD (0), EXIT (1), or TIGHTEN (2).
        realized_pnl_pct: The realized P&L % if exiting (only used for EXIT).
        is_profitable: Whether the position is currently profitable (for TIGHTEN reward).

    Returns:
        Scalar reward value.
    """
    if action == EXIT:
        return realized_pnl_pct
    elif action == HOLD:
        return -0.001
    elif action == TIGHTEN:
        return 0.002 if is_profitable else 0.0
    return 0.0


class RLExitAgent:
    """
    Tabular Q-learning agent for position exit decisions.

    Maintains a Q-table mapping discretized states to action values.
    Uses epsilon-greedy exploration with decay.
    """

    def __init__(
        self,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9,
        epsilon_min: float = 0.01,
        model_name: str = "rl_exit_qtable",
    ):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model_name = model_name
        self.model_path = MODEL_DIR / f"{model_name}.pkl"

        # Q-table: dict mapping (bin_tuple) -> np.array of shape (3,)
        self.q_table: dict[tuple, np.ndarray] = {}

        # Training stats
        self.episodes_trained = 0
        self.total_reward = 0.0

        self._load()

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Get Q-values for a state, initializing to zeros if unseen."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(3, dtype=np.float64)
        return self.q_table[state_key]

    def decide(self, state: PositionState) -> dict:
        """
        Choose an action for the given position state (inference only).
        Always greedy — no exploration during live trading.

        Args:
            state: Current position state.

        Returns:
            Dict with action name, index, Q-values, and confidence.
        """
        state_array = state.to_array()
        state_key = _discretize(state_array)
        q_values = self._get_q_values(state_key)

        action = int(np.argmax(q_values))
        max_q = float(q_values[action])

        # Confidence: how much better the best action is vs. alternatives
        q_range = float(q_values.max() - q_values.min())
        confidence = min(q_range / 0.05, 1.0) if q_range > 0 else 0.0

        return {
            "action": ACTION_NAMES[action],
            "action_idx": action,
            "q_values": {ACTION_NAMES[i]: round(float(q_values[i]), 6) for i in range(3)},
            "confidence": round(confidence, 4),
            "max_q": round(max_q, 6),
            "state_key": state_key,
            "episodes_trained": self.episodes_trained,
        }

    def train_step(
        self,
        state: PositionState,
        action: int,
        reward: float,
        next_state: Optional[PositionState],
        done: bool,
    ) -> float:
        """
        Single Q-learning update step.

        Args:
            state: Current state.
            action: Action taken (0, 1, or 2).
            reward: Reward received.
            next_state: Next state (None if terminal).
            done: Whether the episode is over (position closed).

        Returns:
            The TD error magnitude.
        """
        if action not in (HOLD, EXIT, TIGHTEN):
            log.warning(f"Invalid action: {action}")
            return 0.0

        state_key = _discretize(state.to_array())
        q_values = self._get_q_values(state_key)

        if done or next_state is None:
            td_target = reward
        else:
            next_key = _discretize(next_state.to_array())
            next_q = self._get_q_values(next_key)
            td_target = reward + self.gamma * float(np.max(next_q))

        td_error = td_target - q_values[action]

        # Q-learning update (creates new array, immutable style)
        updated_q = q_values.copy()
        updated_q[action] = q_values[action] + self.lr * td_error
        self.q_table[state_key] = updated_q

        self.total_reward += reward

        return abs(td_error)

    def train(
        self,
        episodes: list[list[dict]],
        verbose: bool = True,
    ) -> dict:
        """
        Train the agent on a batch of episodes.

        Each episode is a list of transition dicts:
            {
                "state": PositionState,
                "action": int,
                "reward": float,
                "next_state": PositionState or None,
                "done": bool,
            }

        Args:
            episodes: List of episodes, each a list of transitions.
            verbose: Whether to log training progress.

        Returns:
            Training summary dict.
        """
        total_td_error = 0.0
        total_steps = 0
        episode_rewards = []

        for ep_idx, episode in enumerate(episodes):
            ep_reward = 0.0

            for transition in episode:
                td_err = self.train_step(
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"],
                    next_state=transition.get("next_state"),
                    done=transition.get("done", False),
                )
                total_td_error += td_err
                total_steps += 1
                ep_reward += transition["reward"]

            episode_rewards.append(ep_reward)
            self.episodes_trained += 1

        # Decay epsilon after training batch
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        avg_td = total_td_error / max(total_steps, 1)
        avg_reward = sum(episode_rewards) / max(len(episode_rewards), 1)

        summary = {
            "episodes_trained": self.episodes_trained,
            "batch_episodes": len(episodes),
            "total_steps": total_steps,
            "avg_td_error": round(avg_td, 6),
            "avg_episode_reward": round(avg_reward, 6),
            "q_table_size": len(self.q_table),
            "epsilon": round(self.epsilon, 4),
        }

        if verbose:
            log.info(
                f"RL Exit trained: {len(episodes)} episodes, "
                f"avg_reward={avg_reward:.4f}, avg_td={avg_td:.6f}, "
                f"q_table={len(self.q_table)} states, eps={self.epsilon:.4f}"
            )

        self.save()
        return summary

    def choose_action_epsilon_greedy(self, state: PositionState) -> int:
        """
        Choose action with epsilon-greedy exploration (for training only).

        Args:
            state: Current position state.

        Returns:
            Action index (0=HOLD, 1=EXIT, 2=TIGHTEN).
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, 3))

        state_key = _discretize(state.to_array())
        q_values = self._get_q_values(state_key)
        return int(np.argmax(q_values))

    def save(self):
        """Persist Q-table and metadata to disk."""
        data = {
            "q_table": self.q_table,
            "episodes_trained": self.episodes_trained,
            "total_reward": self.total_reward,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
        }
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(data, f)
            log.info(f"RL Exit Q-table saved: {len(self.q_table)} states → {self.model_path}")
        except Exception as e:
            log.error(f"Failed to save Q-table: {e}")

    def _load(self):
        """Load Q-table from disk if it exists."""
        if not self.model_path.exists():
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.q_table = data.get("q_table", {})
            self.episodes_trained = data.get("episodes_trained", 0)
            self.total_reward = data.get("total_reward", 0.0)
            self.epsilon = data.get("epsilon", self.epsilon)
            log.info(
                f"RL Exit Q-table loaded: {len(self.q_table)} states, "
                f"{self.episodes_trained} episodes"
            )
        except Exception as e:
            log.warning(f"Failed to load Q-table: {e}")

    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            "q_table_size": len(self.q_table),
            "episodes_trained": self.episodes_trained,
            "total_reward": round(self.total_reward, 4),
            "epsilon": round(self.epsilon, 4),
            "lr": self.lr,
            "gamma": self.gamma,
            "model_path": str(self.model_path),
        }


def build_state_from_position(
    entry_price: float,
    current_price: float,
    bars_held: int,
    max_bars: int,
    premium_prices: list[float],
    stop_loss: float,
    target: float,
    trailing_active: bool,
    peak_price: float,
) -> PositionState:
    """
    Build a PositionState from raw position data.
    Convenience function for integrating with the risk manager.

    Args:
        entry_price: Entry premium price.
        current_price: Current premium price.
        bars_held: Number of bars the position has been open.
        max_bars: Maximum bars for normalization (e.g., 100).
        premium_prices: Recent premium prices (last 10-20 bars).
        stop_loss: Current stop-loss price.
        target: Target price.
        trailing_active: Whether trailing stop is active.
        peak_price: Highest premium seen during this trade.

    Returns:
        PositionState ready for the RL agent.
    """
    if entry_price <= 0:
        entry_price = 1.0  # prevent division by zero

    unrealized_pnl_pct = (current_price - entry_price) / entry_price

    bars_held_norm = min(bars_held / max(max_bars, 1), 1.0)

    # Premium momentum: % change over last 3 bars
    if len(premium_prices) >= 4:
        recent = premium_prices[-4:]
        premium_momentum = (recent[-1] - recent[0]) / max(abs(recent[0]), 0.01)
    else:
        premium_momentum = 0.0

    # Premium volatility: std of recent returns
    if len(premium_prices) >= 3:
        returns = [
            (premium_prices[i] - premium_prices[i - 1]) / max(abs(premium_prices[i - 1]), 0.01)
            for i in range(1, len(premium_prices))
        ]
        premium_volatility = float(np.std(returns)) if returns else 0.0
    else:
        premium_volatility = 0.0

    # Distance to SL: 0 = at SL, 1 = far from SL
    price_range = abs(entry_price - stop_loss) if entry_price != stop_loss else 1.0
    distance_to_sl = min(abs(current_price - stop_loss) / price_range, 1.0) if price_range > 0 else 0.5

    # Distance to target: 0 = at target, 1 = far
    tgt_range = abs(target - entry_price) if target != entry_price else 1.0
    distance_to_tgt = min(abs(target - current_price) / tgt_range, 1.0) if tgt_range > 0 else 0.5

    peak_gain_pct = max((peak_price - entry_price) / entry_price, 0.0)

    return PositionState(
        unrealized_pnl_pct=unrealized_pnl_pct,
        bars_held_norm=bars_held_norm,
        premium_momentum=premium_momentum,
        premium_volatility=premium_volatility,
        distance_to_sl=distance_to_sl,
        distance_to_tgt=distance_to_tgt,
        trailing_active=1.0 if trailing_active else 0.0,
        peak_gain_pct=peak_gain_pct,
    )
