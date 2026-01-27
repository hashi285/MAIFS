"""
Frequency Tool 개선 테스트
GAN 패턴 감지 추가 후 성능 확인
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.frequency_tool import FrequencyAnalysisTool
from src.tools.base_tool import Verdict

def evaluate_frequency_tool(dataset_path: str, max_samples: int = 50):
    """Frequency Tool 평가"""
    tool = FrequencyAnalysisTool()

    # AI 이미지와 자연 이미지 경로
    ai_images = []
    nature_images = []

    dataset = Path(dataset_path)

    # GenImage BigGAN 구조: val/ai와 val/nature
    ai_path = dataset / "ai"
    nature_path = dataset / "nature"

    if ai_path.exists():
        ai_images = list(ai_path.glob("*.png"))[:max_samples]

    if nature_path.exists():
        nature_images = list(nature_path.glob("*.JPEG"))[:max_samples]

    print(f"AI 이미지: {len(ai_images)}개")
    print(f"자연 이미지: {len(nature_images)}개")

    # 평가
    results = {
        "tp": 0,  # AI를 AI로 판정
        "fp": 0,  # 자연을 AI로 판정
        "tn": 0,  # 자연을 자연으로 판정
        "fn": 0,  # AI를 자연으로 판정
        "uncertain": 0
    }

    start_time = time.time()

    # AI 이미지 평가
    print("\nAI 이미지 평가 중...")
    for i, img_path in enumerate(ai_images):
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            result = tool.analyze(image)

            if result.verdict == Verdict.AI_GENERATED:
                results["tp"] += 1
                status = "✓ TP"
            elif result.verdict == Verdict.UNCERTAIN:
                results["uncertain"] += 1
                status = "? UNCERTAIN"
            else:
                results["fn"] += 1
                status = "✗ FN"

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(ai_images)} - 마지막: {status} (score={result.evidence.get('ai_generation_score', 0):.2f})")

        except Exception as e:
            print(f"  오류: {img_path.name} - {e}")
            results["fn"] += 1

    # 자연 이미지 평가
    print("\n자연 이미지 평가 중...")
    for i, img_path in enumerate(nature_images):
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            result = tool.analyze(image)

            if result.verdict == Verdict.AUTHENTIC:
                results["tn"] += 1
                status = "✓ TN"
            elif result.verdict == Verdict.UNCERTAIN:
                results["uncertain"] += 1
                status = "? UNCERTAIN"
            else:
                results["fp"] += 1
                status = "✗ FP"

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(nature_images)} - 마지막: {status} (score={result.evidence.get('ai_generation_score', 0):.2f})")

        except Exception as e:
            print(f"  오류: {img_path.name} - {e}")
            results["fp"] += 1

    elapsed = time.time() - start_time

    # 메트릭 계산
    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    # 결과 출력
    print("\n" + "=" * 60)
    print("Frequency Tool 평가 결과")
    print("=" * 60)
    print(f"총 샘플: {tp + fp + tn + fn + results['uncertain']}")
    print(f"  TP (AI → AI): {tp}")
    print(f"  FP (자연 → AI): {fp}")
    print(f"  TN (자연 → 자연): {tn}")
    print(f"  FN (AI → 자연): {fn}")
    print(f"  UNCERTAIN: {results['uncertain']}")
    print()
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print()
    print(f"총 소요 시간: {elapsed:.1f}초")
    print(f"평균 처리 시간: {elapsed / (tp + fp + tn + fn + results['uncertain']):.3f}초/이미지")

    # 이전 성능과 비교
    print("\n" + "=" * 60)
    print("성능 개선")
    print("=" * 60)
    print("이전:")
    print("  Recall: 0.28, Precision: 0.875, F1: 0.42")
    print()
    print("현재:")
    print(f"  Recall: {recall:.2f}, Precision: {precision:.3f}, F1: {f1:.2f}")
    print()

    if recall > 0.28:
        improvement = (recall - 0.28) / 0.28 * 100
        print(f"✓ Recall 개선: +{improvement:.1f}%")
    else:
        print("✗ Recall 감소")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "results": results,
        "elapsed": elapsed
    }

if __name__ == "__main__":
    dataset_path = "/root/Desktop/MAIFS/datasets/GenImage_subset/BigGAN/val"
    evaluate_frequency_tool(dataset_path, max_samples=50)
