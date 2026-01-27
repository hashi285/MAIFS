"""
Frequency Tool 에러 분석
어떤 이미지가 오판되고, 어떤 특징이 문제인지 확인
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.frequency_tool import FrequencyAnalysisTool
from src.tools.base_tool import Verdict

def analyze_errors(max_samples: int = 50):
    """에러 분석"""
    dataset_path = Path("/root/Desktop/MAIFS/datasets/GenImage_subset/BigGAN/val")

    ai_path = dataset_path / "ai"
    nature_path = dataset_path / "nature"

    ai_images = list(ai_path.glob("*.png"))[:max_samples]
    nature_images = list(nature_path.glob("*.JPEG"))[:max_samples]

    tool = FrequencyAnalysisTool()

    # 분석 결과 수집
    false_positives = []  # 자연 → AI 오판
    false_negatives = []  # AI → 자연 오판

    print("=" * 80)
    print("False Positive 분석 (자연 이미지를 AI로 오판)")
    print("=" * 80)

    for img_path in nature_images:
        image = np.array(Image.open(img_path).convert("RGB"))
        result = tool.analyze(image)

        if result.verdict == Verdict.AI_GENERATED:
            evidence = result.evidence
            false_positives.append({
                "filename": img_path.name,
                "ai_score": evidence["ai_generation_score"],
                "ai_score_raw": evidence["ai_generation_score_raw"],
                "grid_score": evidence["grid_analysis"]["regularity_score"],
                "checkerboard_score": evidence["gan_checkerboard_analysis"]["checkerboard_score"],
                "slope_score": evidence["power_spectrum_slope_analysis"]["slope_score"],
                "hf_abnormality": evidence["high_frequency_analysis"]["abnormality_score"],
                "is_jpeg": evidence["grid_analysis"]["is_likely_jpeg"],
                "jpeg_penalty": evidence["jpeg_penalty"]
            })

    print(f"\nFalse Positives: {len(false_positives)}/{len(nature_images)}")

    if false_positives:
        # 점수 통계
        scores = {
            "ai_score": [fp["ai_score"] for fp in false_positives],
            "grid": [fp["grid_score"] for fp in false_positives],
            "checkerboard": [fp["checkerboard_score"] for fp in false_positives],
            "slope": [fp["slope_score"] for fp in false_positives],
            "hf": [fp["hf_abnormality"] for fp in false_positives]
        }

        print("\n점수 분포:")
        for name, vals in scores.items():
            mean = np.mean(vals)
            median = np.median(vals)
            std = np.std(vals)
            print(f"  {name:15s}: mean={mean:.3f}, median={median:.3f}, std={std:.3f}")

        # 가장 높은 점수 기여자 확인
        print("\n각 특징의 기여도 (평균):")
        total_raw = np.mean([fp["ai_score_raw"] for fp in false_positives])
        grid_contrib = np.mean([fp["grid_score"] for fp in false_positives]) * 0.35
        checker_contrib = np.mean([fp["checkerboard_score"] for fp in false_positives]) * 0.25
        slope_contrib = np.mean([fp["slope_score"] for fp in false_positives]) * 0.20
        hf_contrib = np.mean([fp["hf_abnormality"] for fp in false_positives]) * 0.20

        print(f"  Grid (35%):        {grid_contrib:.3f} / {total_raw:.3f} = {grid_contrib/total_raw*100:.1f}%")
        print(f"  Checkerboard (25%): {checker_contrib:.3f} / {total_raw:.3f} = {checker_contrib/total_raw*100:.1f}%")
        print(f"  Slope (20%):       {slope_contrib:.3f} / {total_raw:.3f} = {slope_contrib/total_raw*100:.1f}%")
        print(f"  HF (20%):          {hf_contrib:.3f} / {total_raw:.3f} = {hf_contrib/total_raw*100:.1f}%")

        # 상위 10개 FP 출력
        print("\n가장 심각한 False Positives (상위 10개):")
        sorted_fp = sorted(false_positives, key=lambda x: x["ai_score"], reverse=True)[:10]
        for i, fp in enumerate(sorted_fp, 1):
            print(f"\n{i}. {fp['filename']}")
            print(f"   AI 점수: {fp['ai_score']:.3f}")
            print(f"   Grid: {fp['grid_score']:.3f}, Checker: {fp['checkerboard_score']:.3f}, "
                  f"Slope: {fp['slope_score']:.3f}, HF: {fp['hf_abnormality']:.3f}")
            print(f"   JPEG: {fp['is_jpeg']}")

    print("\n" + "=" * 80)
    print("False Negative 분석 (AI 이미지를 자연으로 오판)")
    print("=" * 80)

    for img_path in ai_images:
        image = np.array(Image.open(img_path).convert("RGB"))
        result = tool.analyze(image)

        if result.verdict == Verdict.AUTHENTIC:
            evidence = result.evidence
            false_negatives.append({
                "filename": img_path.name,
                "ai_score": evidence["ai_generation_score"],
                "ai_score_raw": evidence["ai_generation_score_raw"],
                "grid_score": evidence["grid_analysis"]["regularity_score"],
                "checkerboard_score": evidence["gan_checkerboard_analysis"]["checkerboard_score"],
                "slope_score": evidence["power_spectrum_slope_analysis"]["slope_score"],
                "hf_abnormality": evidence["high_frequency_analysis"]["abnormality_score"],
                "is_jpeg": evidence["grid_analysis"]["is_likely_jpeg"]
            })

    print(f"\nFalse Negatives: {len(false_negatives)}/{len(ai_images)}")

    if false_negatives:
        scores = {
            "ai_score": [fn["ai_score"] for fn in false_negatives],
            "grid": [fn["grid_score"] for fn in false_negatives],
            "checkerboard": [fn["checkerboard_score"] for fn in false_negatives],
            "slope": [fn["slope_score"] for fn in false_negatives],
            "hf": [fn["hf_abnormality"] for fn in false_negatives]
        }

        print("\n점수 분포:")
        for name, vals in scores.items():
            mean = np.mean(vals)
            median = np.median(vals)
            std = np.std(vals)
            print(f"  {name:15s}: mean={mean:.3f}, median={median:.3f}, std={std:.3f}")

        # 하위 10개 FN 출력
        print("\n가장 심각한 False Negatives (하위 10개):")
        sorted_fn = sorted(false_negatives, key=lambda x: x["ai_score"])[:10]
        for i, fn in enumerate(sorted_fn, 1):
            print(f"\n{i}. {fn['filename']}")
            print(f"   AI 점수: {fn['ai_score']:.3f}")
            print(f"   Grid: {fn['grid_score']:.3f}, Checker: {fn['checkerboard_score']:.3f}, "
                  f"Slope: {fn['slope_score']:.3f}, HF: {fn['hf_abnormality']:.3f}")
            print(f"   JPEG: {fn['is_jpeg']}")

    # 권장사항 출력
    print("\n" + "=" * 80)
    print("권장사항")
    print("=" * 80)

    if false_positives:
        # FP의 평균 특징 점수
        fp_means = {
            "grid": np.mean([fp["grid_score"] for fp in false_positives]),
            "checkerboard": np.mean([fp["checkerboard_score"] for fp in false_positives]),
            "slope": np.mean([fp["slope_score"] for fp in false_positives]),
            "hf": np.mean([fp["hf_abnormality"] for fp in false_positives])
        }

        # 가장 높은 특징 찾기
        max_feature = max(fp_means.items(), key=lambda x: x[1])

        print(f"\n주요 문제: {max_feature[0]} 특징이 자연 이미지에서 과도하게 반응 (평균 {max_feature[1]:.3f})")

        if max_feature[0] == "checkerboard":
            print("  → GAN checkerboard 감지 알고리즘 개선 필요")
            print("  → 또는 checkerboard 가중치 낮추기 (0.25 → 0.15)")
        elif max_feature[0] == "slope":
            print("  → Power spectrum slope 임계값 조정 필요")
            print("  → 또는 slope 가중치 낮추기 (0.20 → 0.10)")
        elif max_feature[0] == "hf":
            print("  → 고주파 abnormality 임계값 조정 필요")

        # 임계값 제안
        fp_ai_scores = [fp["ai_score"] for fp in false_positives]
        suggested_threshold = np.percentile(fp_ai_scores, 90)  # FP의 90%를 걸러내는 임계값

        print(f"\n임계값 제안:")
        print(f"  현재 ai_threshold: 0.48")
        print(f"  제안 ai_threshold: {suggested_threshold:.2f} (FP 90% 제거)")

    # JSON 저장
    output = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "summary": {
            "fp_count": len(false_positives),
            "fn_count": len(false_negatives),
            "total_nature": len(nature_images),
            "total_ai": len(ai_images)
        }
    }

    output_path = Path("/root/Desktop/MAIFS/outputs/frequency_error_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\n분석 결과 저장: {output_path}")

if __name__ == "__main__":
    analyze_errors(max_samples=50)
