"""
vLLM TP/EP 通信劫持 Demo
==========================

功能：在 vLLM 的 Tensor Parallel (TP) 和 Expert Parallel (EP) 通信层插入拦截器，
      当通信负载过高时，选择性丢弃部分通信流，从而减轻网络压力。
      被丢弃的 token 位置会被记录，供上层 Scheduler 做最终 drop 决策。

使用方法：
    1. 在启动 vLLM 之前 import 本模块并调用 `CommInterceptor.enable()`
    2. 通过环境变量或代码配置劫持策略
    3. 在 Scheduler 中查询 `DropRegistry.get_dropped_tokens()` 并 drop 对应 req

作者：AI Assistant
"""

import os
import time
import random
import threading
import warnings
from enum import Enum, auto
from typing import Optional, Callable, List, Set, Dict, Any
from dataclasses import dataclass, field
from functools import wraps
from collections import deque
import logging

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("comm_hijack")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(name)s] %(asctime)s %(levelname)s: %(message)s"
    ))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# 配置与策略
# ---------------------------------------------------------------------------
class HijackStrategy(Enum):
    """劫持策略"""
    RANDOM = auto()          # 随机丢弃一定比例的通信
    LOAD_BASED = auto()      # 基于通信频率/负载动态丢弃
    TOKEN_MASK = auto()      # 对 tensor 的特定 token 位置做 mask
    BYPASS = auto()          # 直接 bypass 所有通信（极端测试用）


@dataclass
class HijackConfig:
    """劫持配置"""
    strategy: HijackStrategy = HijackStrategy.LOAD_BASED

    # --- RANDOM 策略参数 ---
    random_drop_ratio: float = 0.05        # 随机丢弃 5% 的通信

    # --- LOAD_BASED 策略参数 ---
    load_window_size: int = 50             # 滑动窗口大小（记录最近 N 次通信）
    load_threshold: float = 1000.0         # 通信频率阈值（次/秒），超过则触发丢弃
    load_drop_ratio: float = 0.10          # 过载时丢弃 10% 的通信

    # --- TOKEN_MASK 策略参数 ---
    token_mask_ratio: float = 0.05         # 对 token 维度 mask 的比例

    # --- 通用参数 ---
    enabled: bool = True
    verbose: bool = True

    # 被丢弃 token 的回调（供 Scheduler 联动）
    on_tokens_dropped: Optional[Callable[[int, List[int]], None]] = None
    # 签名: on_tokens_dropped(rank: int, token_indices: List[int]) -> None


# ---------------------------------------------------------------------------
# 全局状态：被丢弃 Token 的注册表
# ---------------------------------------------------------------------------
class DropRegistry:
    """
    记录被劫持通信所影响的 token 索引。
    由于通信层（tensor）本身不携带 req_id，这里通过 batch 维度索引来标识。
    Scheduler 可以在每次迭代前读取本注册表，决定哪些 seq_group 需要被 drop。
    """
    _lock = threading.Lock()
    _dropped_tokens: Dict[int, Set[int]] = {}   # rank -> set of token indices
    _drop_history: deque = deque(maxlen=10000)  # (timestamp, rank, token_idx)
    _total_comm_calls: int = 0
    _dropped_comm_calls: int = 0

    @classmethod
    def record_drop(cls, rank: int, token_indices: List[int]):
        with cls._lock:
            if rank not in cls._dropped_tokens:
                cls._dropped_tokens[rank] = set()
            for idx in token_indices:
                cls._dropped_tokens[rank].add(idx)
                cls._drop_history.append((time.time(), rank, idx))
            cls._dropped_comm_calls += 1

    @classmethod
    def get_dropped_tokens(cls, rank: int) -> Set[int]:
        with cls._lock:
            return cls._dropped_tokens.get(rank, set()).copy()

    @classmethod
    def clear_rank(cls, rank: int):
        with cls._lock:
            cls._dropped_tokens.pop(rank, None)

    @classmethod
    def clear_all(cls):
        with cls._lock:
            cls._dropped_tokens.clear()
            cls._drop_history.clear()

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        with cls._lock:
            return {
                "total_comm_calls": cls._total_comm_calls,
                "dropped_comm_calls": cls._dropped_comm_calls,
                "drop_rate": (
                    cls._dropped_comm_calls / max(cls._total_comm_calls, 1)
                ),
                "active_dropped_tokens": {
                    r: len(s) for r, s in cls._dropped_tokens.items()
                },
            }

    @classmethod
    def increment_total(cls):
        with cls._lock:
            cls._total_comm_calls += 1


# ---------------------------------------------------------------------------
# 负载监控器
# ---------------------------------------------------------------------------
class LoadMonitor:
    """基于滑动窗口的通信频率监控"""
    def __init__(self, window_size: int = 50):
        self.window = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self):
        with self._lock:
            self.window.append(time.time())

    def current_load(self) -> float:
        """返回最近窗口内的平均通信频率（次/秒）"""
        with self._lock:
            if len(self.window) < 2:
                return 0.0
            duration = self.window[-1] - self.window[0]
            if duration <= 0:
                return 0.0
            return (len(self.window) - 1) / duration


# ---------------------------------------------------------------------------
# Mock 函数（当 vllm 未安装时使用）
# ---------------------------------------------------------------------------
def _mock_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """Mock vLLM 的 TP AllReduce（无实际通信）"""
    return input_

class _MockTPGroup:
    """Mock vLLM TP Group"""
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_

class _MockEPGroup:
    """Mock vLLM EP Group"""
    def dispatch(self, hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states
    def combine(self, hidden_states: torch.Tensor, *args, **kwargs):
        return hidden_states


# ---------------------------------------------------------------------------
# 核心拦截器
# ---------------------------------------------------------------------------
class CommInterceptor:
    """
    vLLM TP/EP 通信拦截器。

    通过 monkey-patching 以下关键路径实现劫持：
    1. TP AllReduce: `vllm.distributed.communication_op.tensor_model_parallel_all_reduce`
    2. TP Group 底层: `get_tp_group().all_reduce()`
    3. EP Dispatch: `get_ep_group().dispatch()`
    4. EP Combine: `get_ep_group().combine()`
    5. PyTorch 原生: `torch.distributed.all_reduce`（兜底）

    劫持逻辑：
    - 当策略触发时，不执行（或部分执行）原始通信操作
    - 对 TP AllReduce：将被丢弃的 token 位置在 tensor 中置零，这样即使 all_reduce
      后，这些 token 的贡献也为 0，等价于在模型层面被 drop。
    - 对 EP AllToAll：在 dispatch 阶段阻止特定 token 被发送到远程 rank，
      在 combine 阶段阻止接收回来。
    """

    _instance: Optional["CommInterceptor"] = None
    _enabled: bool = False
    _config: HijackConfig = field(default_factory=HijackConfig)
    _load_monitor = LoadMonitor()

    # 原始函数引用（用于恢复）
    _orig_tensor_model_parallel_all_reduce = None
    _orig_tp_group_all_reduce = None
    _orig_ep_group_dispatch = None
    _orig_ep_group_combine = None
    _orig_torch_all_reduce = None
    _orig_torch_all_to_all_single = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[HijackConfig] = None):
        if config is not None:
            self._config = config
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    @classmethod
    def enable(cls, config: Optional[HijackConfig] = None):
        """启用拦截器（应在 vLLM 初始化之前调用）"""
        if cls._enabled:
            logger.warning("CommInterceptor 已经启用，跳过重复初始化")
            return cls._instance

        inst = cls(config or HijackConfig())
        inst._patch()
        cls._enabled = True
        logger.info("CommInterceptor 已启用，策略=%s", inst._config.strategy.name)
        return inst

    @classmethod
    def disable(cls):
        """关闭拦截器，恢复原始函数"""
        if not cls._enabled or cls._instance is None:
            return
        cls._instance._unpatch()
        cls._enabled = False
        logger.info("CommInterceptor 已关闭")

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def get_config(cls) -> HijackConfig:
        if cls._instance is None:
            return HijackConfig()
        return cls._instance._config

    # ------------------------------------------------------------------
    # 劫持决策逻辑
    # ------------------------------------------------------------------
    def _should_hijack(self, tensor: torch.Tensor) -> bool:
        """根据当前策略决定是否劫持本次通信"""
        cfg = self._config
        if not cfg.enabled:
            return False

        if cfg.strategy == HijackStrategy.RANDOM:
            return random.random() < cfg.random_drop_ratio

        elif cfg.strategy == HijackStrategy.LOAD_BASED:
            self._load_monitor.record()
            load = self._load_monitor.current_load()
            if load > cfg.load_threshold:
                if cfg.verbose:
                    logger.debug(
                        "[Rank %d] 通信负载 %.1f ops/s 超过阈值 %.1f，触发丢弃",
                        self._rank, load, cfg.load_threshold
                    )
                return random.random() < cfg.load_drop_ratio
            return False

        elif cfg.strategy == HijackStrategy.TOKEN_MASK:
            # TOKEN_MASK 策略不在通信级别全部丢弃，而是对 tensor 做 mask
            return False  # 由 _apply_token_mask 处理

        elif cfg.strategy == HijackStrategy.BYPASS:
            return True

        return False

    def _get_token_indices_to_drop(self, tensor: torch.Tensor) -> List[int]:
        """
        根据策略计算需要被 drop 的 token 索引（tensor 的第 0 维）。
        返回的是 batch 维度上的索引列表。
        """
        cfg = self._config
        num_tokens = tensor.shape[0] if tensor.dim() > 0 else 1

        if cfg.strategy == HijackStrategy.TOKEN_MASK:
            drop_count = max(1, int(num_tokens * cfg.token_mask_ratio))
            return random.sample(range(num_tokens), min(drop_count, num_tokens))

        elif cfg.strategy in (HijackStrategy.RANDOM, HijackStrategy.LOAD_BASED):
            # 如果已经决定劫持本次通信，随机选一部分 token 置零
            drop_count = max(1, int(num_tokens * 0.2))  # 一次劫持 20% token
            return random.sample(range(num_tokens), min(drop_count, num_tokens))

        return []

    def _apply_token_mask(self, tensor: torch.Tensor, token_indices: List[int]) -> torch.Tensor:
        """
        将 tensor 中指定 token 索引的位置置零。
        这样即使后续执行 all_reduce，这些 token 也不会对结果产生贡献。
        """
        if not token_indices or tensor.numel() == 0:
            return tensor

        masked = tensor.clone()
        try:
            # 假设第 0 维是 token/batch 维度
            for idx in token_indices:
                if idx < masked.shape[0]:
                    masked[idx].zero_()
        except Exception as e:
            logger.warning("mask 失败: %s", e)
        return masked

    def _notify_drop(self, token_indices: List[int]):
        """通知上层哪些 token 被 drop 了"""
        if not token_indices:
            return
        DropRegistry.record_drop(self._rank, token_indices)
        cfg = self._config
        if cfg.on_tokens_dropped:
            try:
                cfg.on_tokens_dropped(self._rank, token_indices)
            except Exception as e:
                logger.error("on_tokens_dropped 回调出错: %s", e)
        if cfg.verbose:
            logger.info(
                "[Rank %d] 已劫持通信，drop tokens: %s",
                self._rank, token_indices
            )

    # ------------------------------------------------------------------
    # Monkey Patching
    # ------------------------------------------------------------------
    def _patch(self):
        """注入劫持逻辑到 vLLM / PyTorch 通信层"""
        # 1. 尝试 patch vLLM 的 TP all_reduce 高层接口
        try:
            import vllm.distributed.communication_op as comm_op
            self._orig_tensor_model_parallel_all_reduce = comm_op.tensor_model_parallel_all_reduce
            comm_op.tensor_model_parallel_all_reduce = self._wrapped_tensor_model_parallel_all_reduce
            logger.debug("Patched vllm.distributed.communication_op.tensor_model_parallel_all_reduce")
        except ImportError as e:
            logger.warning("无法导入 vllm.distributed.communication_op: %s", e)
            # 使用 mock 函数兜底，确保 demo 能运行
            self._orig_tensor_model_parallel_all_reduce = _mock_tensor_model_parallel_all_reduce

        # 2. 尝试 patch vLLM 的 TP Group 底层
        try:
            from vllm.distributed.parallel_state import get_tp_group
            tp_group = get_tp_group()
            self._orig_tp_group_all_reduce = tp_group.all_reduce
            tp_group.all_reduce = self._wrapped_tp_group_all_reduce.__get__(tp_group, type(tp_group))
            logger.debug("Patched get_tp_group().all_reduce")
        except Exception as e:
            logger.warning("无法 patch TP group: %s", e)
            self._orig_tp_group_all_reduce = _MockTPGroup().all_reduce

        # 3. 尝试 patch vLLM 的 EP Group dispatch / combine
        try:
            from vllm.distributed.parallel_state import get_ep_group
            ep_group = get_ep_group()
            if hasattr(ep_group, "dispatch"):
                self._orig_ep_group_dispatch = ep_group.dispatch
                ep_group.dispatch = self._wrapped_ep_dispatch.__get__(ep_group, type(ep_group))
                logger.debug("Patched get_ep_group().dispatch")
            if hasattr(ep_group, "combine"):
                self._orig_ep_group_combine = ep_group.combine
                ep_group.combine = self._wrapped_ep_combine.__get__(ep_group, type(ep_group))
                logger.debug("Patched get_ep_group().combine")
        except Exception as e:
            logger.warning("无法 patch EP group: %s", e)
            self._orig_ep_group_dispatch = _MockEPGroup().dispatch
            self._orig_ep_group_combine = _MockEPGroup().combine

        # 4. 兜底：patch PyTorch 原生的 all_reduce / all_to_all_single
        self._orig_torch_all_reduce = dist.all_reduce
        dist.all_reduce = self._wrapped_torch_all_reduce

        if hasattr(dist, "all_to_all_single"):
            self._orig_torch_all_to_all_single = dist.all_to_all_single
            dist.all_to_all_single = self._wrapped_torch_all_to_all_single

        logger.info("所有通信层 patch 完成")

    def _unpatch(self):
        """恢复原始函数"""
        if self._orig_tensor_model_parallel_all_reduce:
            try:
                import vllm.distributed.communication_op as comm_op
                comm_op.tensor_model_parallel_all_reduce = self._orig_tensor_model_parallel_all_reduce
            except ImportError:
                pass

        if self._orig_tp_group_all_reduce:
            try:
                from vllm.distributed.parallel_state import get_tp_group
                get_tp_group().all_reduce = self._orig_tp_group_all_reduce
            except Exception:
                pass

        if self._orig_ep_group_dispatch:
            try:
                from vllm.distributed.parallel_state import get_ep_group
                get_ep_group().dispatch = self._orig_ep_group_dispatch
            except Exception:
                pass

        if self._orig_ep_group_combine:
            try:
                from vllm.distributed.parallel_state import get_ep_group
                get_ep_group().combine = self._orig_ep_group_combine
            except Exception:
                pass

        if self._orig_torch_all_reduce:
            dist.all_reduce = self._orig_torch_all_reduce
        if self._orig_torch_all_to_all_single:
            dist.all_to_all_single = self._orig_torch_all_to_all_single

        logger.info("所有通信层已恢复")

    # ------------------------------------------------------------------
    # Wrapper 函数
    # ------------------------------------------------------------------
    def _wrapped_tensor_model_parallel_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """劫持 vLLM 高层的 TP AllReduce"""
        DropRegistry.increment_total()

        if self._should_hijack(input_):
            token_indices = self._get_token_indices_to_drop(input_)
            masked = self._apply_token_mask(input_, token_indices)
            self._notify_drop(token_indices)
            # 仍然执行 all_reduce，但被 mask 的 token 贡献为 0
            return self._orig_tensor_model_parallel_all_reduce(masked)

        return self._orig_tensor_model_parallel_all_reduce(input_)

    def _wrapped_tp_group_all_reduce(self, *args, **kwargs):
        """劫持 TP Group 底层的 all_reduce"""
        DropRegistry.increment_total()
        # 这里 args[0] 通常是 tensor
        tensor = args[0] if args else kwargs.get("input_")

        if tensor is not None and self._should_hijack(tensor):
            token_indices = self._get_token_indices_to_drop(tensor)
            if args:
                args = (self._apply_token_mask(args[0], token_indices),) + args[1:]
            else:
                kwargs["input_"] = self._apply_token_mask(tensor, token_indices)
            self._notify_drop(token_indices)

        return self._orig_tp_group_all_reduce(*args, **kwargs)

    def _wrapped_ep_dispatch(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        劫持 EP 的 dispatch（token 分发到专家所在 rank）。
        如果被劫持，阻止部分 token 被发送出去。
        """
        DropRegistry.increment_total()

        if self._should_hijack(hidden_states):
            token_indices = self._get_token_indices_to_drop(hidden_states)
            masked = self._apply_token_mask(hidden_states, token_indices)
            self._notify_drop(token_indices)
            return self._orig_ep_group_dispatch(masked, *args, **kwargs)

        return self._orig_ep_group_dispatch(hidden_states, *args, **kwargs)

    def _wrapped_ep_combine(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        劫持 EP 的 combine（专家计算结果回收）。
        如果被劫持，对回收后的结果中特定 token 置零。
        """
        DropRegistry.increment_total()

        result = self._orig_ep_group_combine(hidden_states, *args, **kwargs)

        if self._should_hijack(result if isinstance(result, torch.Tensor) else hidden_states):
            token_indices = self._get_token_indices_to_drop(
                result if isinstance(result, torch.Tensor) else hidden_states
            )
            if isinstance(result, torch.Tensor):
                result = self._apply_token_mask(result, token_indices)
            self._notify_drop(token_indices)

        return result

    def _wrapped_torch_all_reduce(self, tensor, *args, **kwargs):
        """兜底：劫持 PyTorch 原生的 all_reduce"""
        DropRegistry.increment_total()

        if self._should_hijack(tensor):
            token_indices = self._get_token_indices_to_drop(tensor)
            self._apply_token_mask(tensor, token_indices)  # in-place
            self._notify_drop(token_indices)

        return self._orig_torch_all_reduce(tensor, *args, **kwargs)

    def _wrapped_torch_all_to_all_single(self, output, input, *args, **kwargs):
        """兜底：劫持 PyTorch 原生的 all_to_all_single"""
        DropRegistry.increment_total()

        if self._should_hijack(input):
            token_indices = self._get_token_indices_to_drop(input)
            masked = self._apply_token_mask(input, token_indices)
            self._notify_drop(token_indices)
            return self._orig_torch_all_to_all_single(output, masked, *args, **kwargs)

        return self._orig_torch_all_to_all_single(output, input, *args, **kwargs)


# ---------------------------------------------------------------------------
# 与 vLLM Scheduler 的联动示例
# ---------------------------------------------------------------------------
class SchedulerDropAdapter:
    """
    供 vLLM Scheduler 调用的适配器。

    使用方式（在 Scheduler 的 schedule() 方法中）：

        from comm_hijack_demo import SchedulerDropAdapter

        # 每次调度前，获取被通信层标记为 drop 的 token 索引
        dropped = SchedulerDropAdapter.get_dropped_token_indices(rank)

        # 将 token 索引映射到 seq_group（需要用户根据实际 batch 结构实现）
        for sg in running:
            token_start = sg.metadata.token_offset
            token_end = token_start + len(sg.seq_ids)
            if any(token_start <= idx < token_end for idx in dropped):
                # 标记该 seq_group 为待 drop
                sg.mark_for_drop()
    """

    @staticmethod
    def get_dropped_token_indices(rank: int) -> Set[int]:
        return DropRegistry.get_dropped_tokens(rank)

    @staticmethod
    def clear_dropped_tokens(rank: int):
        DropRegistry.clear_rank(rank)

    @staticmethod
    def get_comm_stats() -> Dict[str, Any]:
        return DropRegistry.get_stats()


# ---------------------------------------------------------------------------
# 模拟测试（无需多 GPU，单进程即可验证逻辑）
# ---------------------------------------------------------------------------
def demo_single_process():
    """单进程 Demo：验证劫持逻辑是否正确工作"""
    print("=" * 60)
    print("单进程 Demo：验证 CommInterceptor 劫持逻辑")
    print("=" * 60)

    # 配置：使用 RANDOM 策略，丢弃率 30%（方便观察）
    cfg = HijackConfig(
        strategy=HijackStrategy.RANDOM,
        random_drop_ratio=0.30,
        verbose=True,
        on_tokens_dropped=lambda rank, indices: print(
            f"  >>> [Callback] Rank {rank} dropped tokens: {indices}"
        )
    )

    # 启用拦截器
    interceptor = CommInterceptor.enable(cfg)

    # 模拟 20 次通信
    for i in range(20):
        tensor = torch.randn(10, 128)
        # 手动触发被劫持的函数（模拟 vLLM 内部调用）
        if hasattr(interceptor, '_wrapped_tensor_model_parallel_all_reduce'):
            _ = interceptor._wrapped_tensor_model_parallel_all_reduce(tensor)

    print("\n--- 统计信息 ---")
    stats = DropRegistry.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 关闭拦截器
    CommInterceptor.disable()
    print("\nDemo 结束，拦截器已关闭")


def demo_with_vllm_mock():
    """
    演示如何在模拟的 vLLM 环境中使用。
    这里通过动态创建 mock 模块来验证 patch 是否成功。
    """
    print("=" * 60)
    print("Mock vLLM 集成 Demo")
    print("=" * 60)

    # 动态创建 mock vllm 模块结构
    import sys
    import types

    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        vllm.distributed = types.ModuleType("vllm.distributed")
        vllm.distributed.communication_op = types.ModuleType("vllm.distributed.communication_op")
        vllm.distributed.parallel_state = types.ModuleType("vllm.distributed.parallel_state")

        # Mock 通信函数
        def mock_tensor_model_parallel_all_reduce(input_):
            print(f"  [Mock vLLM] tensor_model_parallel_all_reduce called, shape={input_.shape}")
            return input_

        class MockGroup:
            def all_reduce(self, input_):
                print(f"  [Mock vLLM] TPGroup.all_reduce called, shape={input_.shape}")
                return input_
            def dispatch(self, hidden_states, *args, **kwargs):
                print(f"  [Mock vLLM] EPGroup.dispatch called, shape={hidden_states.shape}")
                return hidden_states
            def combine(self, hidden_states, *args, **kwargs):
                print(f"  [Mock vLLM] EPGroup.combine called, shape={hidden_states.shape}")
                return hidden_states

        vllm.distributed.communication_op.tensor_model_parallel_all_reduce = mock_tensor_model_parallel_all_reduce
        vllm.distributed.parallel_state.get_tp_group = MockGroup
        vllm.distributed.parallel_state.get_ep_group = MockGroup

        sys.modules["vllm"] = vllm
        sys.modules["vllm.distributed"] = vllm.distributed
        sys.modules["vllm.distributed.communication_op"] = vllm.distributed.communication_op
        sys.modules["vllm.distributed.parallel_state"] = vllm.distributed.parallel_state

    # 启用劫持
    cfg = HijackConfig(
        strategy=HijackStrategy.LOAD_BASED,
        load_threshold=5.0,   # 低阈值方便触发
        load_drop_ratio=0.5,
        verbose=True
    )
    CommInterceptor.enable(cfg)

    # 模拟快速通信（触发负载阈值）
    import time
    for i in range(20):
        tensor = torch.randn(8, 64)
        # 通过 mock 模块调用
        import vllm.distributed.communication_op as comm_op
        comm_op.tensor_model_parallel_all_reduce(tensor)
        time.sleep(0.01)  # 10ms，快速累积负载

    print("\n--- 统计信息 ---")
    stats = DropRegistry.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    CommInterceptor.disable()


# ---------------------------------------------------------------------------
# 实际 vLLM 集成示例（供复制粘贴）
# ---------------------------------------------------------------------------
"""
# 在启动 vLLM 的脚本中（如 serve.py 或自定义入口），在 LLMEngine 初始化之前加入：

from comm_hijack_demo import CommInterceptor, HijackConfig, HijackStrategy

# 方式 1：环境变量配置（推荐）
import os
os.environ["COMM_HIJACK_STRATEGY"] = "LOAD_BASED"
os.environ["COMM_HIJACK_LOAD_THRESHOLD"] = "800.0"
os.environ["COMM_HIJACK_LOAD_DROP_RATIO"] = "0.15"

# 方式 2：代码配置
cfg = HijackConfig(
    strategy=HijackStrategy.LOAD_BASED,
    load_threshold=800.0,
    load_drop_ratio=0.15,
    on_tokens_dropped=lambda rank, indices: print(f"Rank {rank} drop tokens: {indices}")
)
CommInterceptor.enable(cfg)

# 然后正常启动 vLLM
from vllm import LLM
llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite", tensor_parallel_size=2, enable_expert_parallel=True)

# ---------------------------------------------------------
# 在 Scheduler 中（如 vllm/core/scheduler.py）:
# ---------------------------------------------------------
from comm_hijack_demo import SchedulerDropAdapter

class MyScheduler:
    def schedule(self):
        # ... 原有调度逻辑 ...

        # 1. 获取通信层标记的坏 token
        dropped_tokens = SchedulerDropAdapter.get_dropped_token_indices(self.rank)

        # 2. 映射到 seq_group 并 drop
        for seq_group in running:
            # 假设 batch 中 token 是连续排列的
            if seq_group.token_start_idx in dropped_tokens:
                seq_group.status = SequenceStatus.FINISHED_DROPPED

        # 3. 清理已处理的标记
        SchedulerDropAdapter.clear_dropped_tokens(self.rank)

        # ... 继续原有调度逻辑 ...
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("vLLM TP/EP 通信劫持 Demo")
    print("=" * 60 + "\n")

    # 运行单进程验证
    demo_single_process()

    print("\n")

    # 运行 Mock vLLM 集成验证
    demo_with_vllm_mock()

    print("\n" + "=" * 60)
    print("所有 Demo 完成。请查看上方输出中的劫持日志和统计信息。")
    print("=" * 60)