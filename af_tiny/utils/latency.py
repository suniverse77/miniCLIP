"""모델 효율성 측정 도구."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class LatencyConfig:
    warmup_iters: int = 1000
    measure_iters: int = 1000
    repeat: int = 1


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_inference(model: torch.nn.Module, batch: Tuple[Any, ...]) -> None:
    """CLIP 모델 추론 실행. batch는 dataloader에서 나온 튜플 (img, label, gt, ...)."""
    model.eval()
    with torch.no_grad():
        # dataloader에서 나온 튜플에서 이미지만 추출
        if isinstance(batch, (tuple, list)):
            imgs = batch[0]  # 첫 번째 요소가 이미지
        else:
            imgs = batch
        
        # 모델의 device로 이미지 이동 (evaluation.py와 동일)
        device = _infer_model_device(model)
        imgs = imgs.to(device)
        
        # CLIP 모델의 detect_forward 사용 (evaluation과 동일)
        model.detect_forward(imgs)


def measure_latency(
    model: torch.nn.Module,
    inputs: Iterable[Tuple[Any, ...]],
    *,
    warmup_iters: int = 1000,
    measure_iters: int = 1000,
) -> Dict[str, float]:
    """Latency(ms) 및 FPS 측정."""
    timings = []
    iterator = iter(inputs)

    # Warm-up
    for _ in range(warmup_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        _run_inference(model, batch)
    
    # Warm-up 후 동기화 (정확한 측정을 위해)
    _synchronize()

    for _ in range(measure_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)

        start = time.perf_counter()
        _run_inference(model, batch)
        _synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0  # ms
        timings.append(elapsed)

    timings_tensor = torch.tensor(timings)
    mean = float(timings_tensor.mean())
    std = float(timings_tensor.std(unbiased=False))
    fps = 1000.0 / mean if mean > 0 else float("inf")
    return {"latency_ms_mean": mean, "latency_ms_std": std, "fps": fps}


def measure_throughput(
    model: torch.nn.Module,
    inputs: Iterable[Tuple[Any, ...]],
    *,
    num_passes: int = 1000,
) -> Dict[str, float]:
    """
    Throughput 측정.
    
    Args:
        model: 추론할 모델
        inputs: 입력 데이터 iterator (batch size 16)
        num_passes: 측정할 forward pass 횟수 (기본값: 1000)
    
    Returns:
        throughput: Throughput (samples/sec). Throughput = (batch_size * num_passes) / total_time
    """
    iterator = iter(inputs)
    model.eval()
    
    # Warm-up (100번)
    for _ in range(100):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        _run_inference(model, batch)
    
    _synchronize()
    
    # 1000 pass의 총 시간 측정
    start_time = time.perf_counter()
    total_samples = 0
    
    for _ in range(num_passes):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        
        # 배치 크기 확인 (튜플의 첫 번째 요소가 이미지 텐서)
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            imgs = batch[0]
            if isinstance(imgs, torch.Tensor):
                batch_size = imgs.shape[0]
            else:
                batch_size = 1
        else:
            batch_size = 1
        
        _run_inference(model, batch)
        total_samples += batch_size
    
    _synchronize()
    total_time = time.perf_counter() - start_time  # seconds
    
    # Throughput = total_samples / total_time
    throughput = float(total_samples / total_time) if total_time > 0 else 0.0
    
    return {
        "throughput_samples_per_sec": throughput,
        "throughput_total_samples": float(total_samples),
        "throughput_total_time_sec": total_time,
    }


def measure_gpu_memory(
    model: torch.nn.Module,
    inputs: Iterable[Tuple[Any, ...]],
    *,
    num_iters: int = 1000,
) -> Dict[str, float]:
    """
    GPU Memory 측정 (논문 기준: PyTorch profiler로 peak reserved GPU memory).
    
    Args:
        model: 측정할 모델
        inputs: 입력 데이터 iterator (batch size 1)
        num_iters: 측정 횟수 (기본값: 1000)
    
    Returns:
        peak_reserved_bytes: Peak reserved GPU memory (bytes), 1000회 평균
    """
    if not torch.cuda.is_available():
        return {"gpu_peak_reserved_bytes": float("nan")}
    
    iterator = iter(inputs)
    model.eval()
    
    # Warm-up (100회)
    for _ in range(100):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        _run_inference(model, batch)
    
    _synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 측정 시작
    memory_readings = []
    
    for _ in range(num_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(inputs)
            batch = next(iterator)
        
        torch.cuda.reset_peak_memory_stats()
        _run_inference(model, batch)
        _synchronize()
        
        # Peak reserved memory 측정
        peak_reserved = torch.cuda.max_memory_reserved()
        memory_readings.append(float(peak_reserved))
    
    # 평균 계산
    if memory_readings:
        memory_tensor = torch.tensor(memory_readings)
        avg_peak_reserved = float(memory_tensor.mean())
        std_peak_reserved = float(memory_tensor.std(unbiased=False))
    else:
        avg_peak_reserved = float("nan")
        std_peak_reserved = float("nan")
    
    return {
        "gpu_peak_reserved_bytes": avg_peak_reserved,
        "gpu_peak_reserved_bytes_std": std_peak_reserved,
    }


def measure_params(model: torch.nn.Module) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": float(total_params),
        "params_trainable": float(trainable_params),
    }




def _infer_model_device(model: torch.nn.Module) -> torch.device:
    """모델의 device를 추론."""
    if hasattr(model, "device"):
        device_attr = getattr(model, "device")
        if isinstance(device_attr, torch.device):
            return device_attr
        if isinstance(device_attr, str):
            return torch.device(device_attr)

    for param in model.parameters():
        return param.device

    for buffer in model.buffers():
        return buffer.device

    return torch.device("cpu")


def profile_model(
    model: torch.nn.Module,
    inputs: Iterable[Tuple[Any, ...]],
    *,
    latency_cfg: Optional[LatencyConfig] = None,
    throughput_inputs: Optional[Iterable[Tuple[Any, ...]]] = None,
    enable_gpu_memory_measurement: bool = True,
    throughput_num_passes: int = 1000,
    gpu_memory_num_iters: int = 1000,
) -> Dict[str, float]:
    """
    Latency / Throughput / FPS / GPU Memory / Params 종합 측정.
    
    Args:
        model: 측정할 모델
        inputs: Latency 측정용 입력 (dataloader)
        latency_cfg: Latency 측정 설정
        throughput_inputs: Throughput 측정용 입력. None이면 throughput 측정 안함.
        enable_gpu_memory_measurement: GPU 메모리 측정 여부
        throughput_num_passes: Throughput 측정 횟수 (기본값: 1000)
        gpu_memory_num_iters: GPU 메모리 측정 횟수 (기본값: 1000)
    """
    cfg = latency_cfg or LatencyConfig()
    results: Dict[str, float] = {}

    latency_stats = measure_latency(
        model,
        inputs,
        warmup_iters=cfg.warmup_iters,
        measure_iters=cfg.measure_iters,
    )
    results.update(latency_stats)
    
    # Throughput 측정
    if throughput_inputs is not None:
        throughput_stats = measure_throughput(
            model,
            throughput_inputs,
            num_passes=throughput_num_passes,
        )
        results.update(throughput_stats)
    
    # GPU Memory 측정
    if enable_gpu_memory_measurement and torch.cuda.is_available():
        memory_inputs = inputs
        gpu_memory_stats = measure_gpu_memory(
            model,
            memory_inputs,
            num_iters=gpu_memory_num_iters,
        )
        results.update(gpu_memory_stats)
    
    results.update(measure_params(model))

    # 기존 allocated memory도 유지 (하위 호환성)
    if torch.cuda.is_available():
        results["vram_peak_bytes"] = float(torch.cuda.max_memory_allocated())

    return results


__all__ = [
    "LatencyConfig",
    "measure_latency",
    "measure_throughput",
    "measure_gpu_memory",
    "measure_params",
    "profile_model",
]

