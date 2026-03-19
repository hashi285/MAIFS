"""
에이전트 순차 vs 병렬 처리 inference latency 벤치마크

현재 MAIFS._collect_agent_analyses()는 순차 for 루프.
ThreadPoolExecutor 기반 병렬 처리와 비교하여 실제 속도 차이를 측정.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from PIL import Image


# ── 에이전트 로드 ─────────────────────────────────────────────
from src.maifs import MAIFS

IMAGE_DIR = Path("datasets/CASIA2_subset/Tp")
N_IMAGES = 5       # 테스트할 이미지 수
N_REPEAT = 3       # 반복 횟수 (모델 로딩 후 warm 상태에서)


def load_test_images(n=N_IMAGES):
    paths = list(IMAGE_DIR.glob("*.jpg"))[:n]
    images = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        images.append((p.name, img))
    return images


# ── 순차 처리 ────────────────────────────────────────────────
def analyze_sequential(agents, image):
    responses = {}
    for name, agent in agents.items():
        try:
            responses[name] = agent.analyze(image)
        except Exception as e:
            responses[name] = None
    return responses


# ── 병렬 처리 (ThreadPoolExecutor) ──────────────────────────
def analyze_parallel(agents, image, max_workers=4):
    responses = {}

    def _run(name_agent):
        name, agent = name_agent
        try:
            return name, agent.analyze(image)
        except Exception as e:
            return name, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, item): item[0]
                   for item in agents.items()}
        for future in as_completed(futures):
            name, response = future.result()
            responses[name] = response

    return responses


def measure(fn, agents, images, n_repeat):
    """이미지별 × repeat 측정 → 이미지당 평균 ms 반환"""
    times = []
    for _, img in images:
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(agents, img)
            times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.array(times)


def report(label, arr_ms):
    print(f"  {label}:")
    print(f"    mean   = {arr_ms.mean():.1f} ms")
    print(f"    median = {np.median(arr_ms):.1f} ms")
    print(f"    min    = {arr_ms.min():.1f} ms")
    print(f"    max    = {arr_ms.max():.1f} ms")
    return {"mean": float(arr_ms.mean()), "median": float(np.median(arr_ms)),
            "min": float(arr_ms.min()), "max": float(arr_ms.max())}


def main():
    print("=== Agent Sequential vs Parallel Benchmark ===\n")

    # MAIFS 초기화 (모델 로딩)
    print("모델 로딩 중 (최초 1회)...")
    t0 = time.perf_counter()
    maifs = MAIFS()
    load_time = (time.perf_counter() - t0) * 1000
    print(f"모델 로딩 완료: {load_time:.0f} ms\n")
    print(f"에이전트: {list(maifs.agents.keys())}\n")

    # 이미지 로드
    images = load_test_images()
    print(f"테스트 이미지: {len(images)}장 × {N_REPEAT}회 반복\n")

    # warmup (첫 번째 이미지로 워밍업)
    print("Warmup 중...")
    _, warmup_img = images[0]
    analyze_sequential(maifs.agents, warmup_img)
    analyze_parallel(maifs.agents, warmup_img)
    print("Warmup 완료\n")

    # ── 순차 측정 ──
    print("[순차 처리 (현재 방식)]")
    seq_times = measure(analyze_sequential, maifs.agents, images, N_REPEAT)
    seq_stats = report("이미지당 전체 에이전트 시간", seq_times)

    # 에이전트별 개별 시간 측정
    print("\n  [에이전트별 단독 시간]")
    agent_stats = {}
    for name, agent in maifs.agents.items():
        times = []
        for _, img in images:
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                try:
                    agent.analyze(img)
                except Exception:
                    pass
                times.append((time.perf_counter() - t0) * 1000)
        mean_t = np.mean(times)
        agent_stats[name] = mean_t
        print(f"    {name:12s}: {mean_t:.1f} ms (mean)")

    # ── 병렬 측정 ──
    print("\n[병렬 처리 (ThreadPoolExecutor, 4 workers)]")
    par_times = measure(analyze_parallel, maifs.agents, images, N_REPEAT)
    par_stats = report("이미지당 전체 에이전트 시간", par_times)

    # ── 요약 ──────────────────────────────────────────────────
    speedup = seq_stats['mean'] / par_stats['mean'] if par_stats['mean'] > 0 else 0
    theoretical = max(agent_stats.values())  # 이론적 병렬 최솟값 = 가장 느린 에이전트

    print("\n" + "=" * 55)
    print("요약")
    print("=" * 55)
    print(f"  모델 로딩 시간        : {load_time:.0f} ms (최초 1회)")
    print(f"  순차 처리 (mean)      : {seq_stats['mean']:.1f} ms/image")
    print(f"  병렬 처리 (mean)      : {par_stats['mean']:.1f} ms/image")
    print(f"  실제 속도 향상        : {speedup:.2f}x")
    print(f"  이론적 최소 (slowest) : {theoretical:.1f} ms ({max(agent_stats, key=agent_stats.get)})")
    print(f"  합의 계층 (DAAC-GBM)  : ~0.07 ms")
    print(f"  병렬 시 총 처리 시간  : {par_stats['mean'] + 0.07:.1f} ms/image")

    # JSON 저장
    result = {
        "model_load_ms": load_time,
        "n_images": len(images),
        "n_repeat": N_REPEAT,
        "sequential_ms": seq_stats,
        "parallel_ms": par_stats,
        "per_agent_ms": agent_stats,
        "speedup": speedup,
        "theoretical_min_ms": theoretical,
    }
    out = Path("experiments/results/agent_parallel_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
