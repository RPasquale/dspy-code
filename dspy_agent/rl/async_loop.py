"""Asynchronous rollout → judge → learner pipeline for toolchain RL.

This is intentionally lightweight (Python threads + queues) so it can run in
restricted environments while demonstrating the systems patterns highlighted in
recent reasoning-RL work (AReaL/Magistral).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .rlkit import RLToolEnv, ToolAction, make_bandit, BaseBandit


@dataclass
class RolloutPacket:
    env: RLToolEnv
    obs: List[float]
    action_index: int
    step_id: int


@dataclass
class JudgedPacket:
    env: RLToolEnv
    obs: List[float]
    next_obs: List[float]
    reward: float
    action_index: int
    info: Dict[str, object]
    step_id: int


@dataclass
class AsyncStats:
    rewards: List[float] = field(default_factory=list)
    processed: int = 0
    last_log: float = field(default_factory=time.time)

    def add(self, reward: float) -> None:
        self.rewards.append(reward)
        if len(self.rewards) > 1000:
            self.rewards.pop(0)
        self.processed += 1

    def summary(self) -> Dict[str, float]:
        avg = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
        return {"count": float(self.processed), "avg_reward": avg}


class AsyncRLTrainer:
    """Coordinate asynchronous rollout, judging, and learner updates."""

    def __init__(
        self,
        make_env: Callable[[], RLToolEnv],
        *,
        policy: str = "epsilon-greedy",
        policy_kwargs: Optional[Dict[str, float]] = None,
        rollout_workers: int = 2,
        judge_workers: int = 2,
        max_queue: int = 64,
        log_interval: float = 10.0,
    ) -> None:
        self.make_env = make_env
        self.policy_name = policy
        self.policy_kwargs = dict(policy_kwargs or {})
        self.rollout_workers = rollout_workers
        self.judge_workers = judge_workers
        self.max_queue = max_queue
        self.log_interval = log_interval

        self._policy: Optional[BaseBandit] = None
        self._policy_lock = threading.Lock()

        self._rollout_queue: "queue.Queue[RLToolEnv]" = queue.Queue(max_queue)
        self._judge_queue: "queue.Queue[RolloutPacket]" = queue.Queue(max_queue)
        self._learn_queue: "queue.Queue[JudgedPacket]" = queue.Queue(max_queue)

        self._stop = threading.Event()
        self._stats = AsyncStats()
        self._step_counter = 0

    # Public API -----------------------------------------------------------

    def start(self) -> None:
        if self._policy is None:
            env = self.make_env()
            self._policy = make_bandit(self.policy_name, env.action_dim, **self.policy_kwargs)
        self._spawn_threads()

    def stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        for t in self._threads:
            t.join()

    def snapshot_stats(self) -> Dict[str, float]:
        return self._stats.summary()

    # Internals ------------------------------------------------------------

    def _spawn_threads(self) -> None:
        self._threads: List[threading.Thread] = []
        for _ in range(self.rollout_workers):
            self._rollout_queue.put(self.make_env())
        for idx in range(self.rollout_workers):
            t = threading.Thread(target=self._rollout_loop, name=f"rollout-{idx}", daemon=True)
            t.start(); self._threads.append(t)
        for idx in range(self.judge_workers):
            t = threading.Thread(target=self._judge_loop, name=f"judge-{idx}", daemon=True)
            t.start(); self._threads.append(t)
        learner = threading.Thread(target=self._learn_loop, name="learner", daemon=True)
        learner.start(); self._threads.append(learner)

    def _next_step_id(self) -> int:
        with self._policy_lock:
            self._step_counter += 1
            return self._step_counter

    def _rollout_loop(self) -> None:
        while not self._stop.is_set():
            try:
                env = self._rollout_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                obs, _ = env.reset()
                with self._policy_lock:
                    assert self._policy is not None
                    action_index = int(self._policy.select(obs))
                pkt = RolloutPacket(env=env, obs=obs, action_index=action_index, step_id=self._next_step_id())
                self._judge_queue.put(pkt)
            except Exception:
                time.sleep(0.1)

    def _judge_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pkt = self._judge_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            env = pkt.env
            try:
                next_obs, reward, terminated, truncated, info = env.step(pkt.action_index)
            except Exception as exc:
                next_obs = pkt.obs
                reward = 0.0
                info = {"error": str(exc)}
                terminated = True
                truncated = True
            judged = JudgedPacket(
                env=env,
                obs=pkt.obs,
                next_obs=next_obs,
                reward=float(reward),
                action_index=pkt.action_index,
                info=info,
                step_id=pkt.step_id,
            )
            self._learn_queue.put(judged)
            try:
                self._rollout_queue.put(env, timeout=0.1)
            except queue.Full:
                pass
            if terminated or truncated:
                try:
                    env.reset()
                except Exception:
                    pass

    def _learn_loop(self) -> None:
        last_log = time.time()
        while not self._stop.is_set():
            try:
                judged = self._learn_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            with self._policy_lock:
                assert self._policy is not None
                self._policy.update(judged.action_index, judged.reward, judged.next_obs)
            self._stats.add(judged.reward)
            now = time.time()
            if now - last_log >= self.log_interval:
                summary = self._stats.summary()
                print(f"[async-rl] steps={summary['count']:.0f} avg_reward={summary['avg_reward']:.3f}")
                last_log = now
