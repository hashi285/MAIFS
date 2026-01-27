"""
Tool Specialization Analysis
각 툴이 어떤 유형의 이미지에서 강점을 가지는지 분석
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.frequency_tool import FrequencyAnalysisTool
from src.tools.noise_tool import NoiseAnalysisTool
from src.tools.base_tool import Verdict

def analyze_specialization():
    """각 툴의 전문 영역 분석"""

    dataset_path = Path("/root/Desktop/MAIFS/datasets/GenImage_subset/BigGAN/val")
    ai_images = list((dataset_path / "ai").glob("*.png"))[:50]
    nature_images = list((dataset_path / "nature").glob("*.JPEG"))[:50]

    freq_tool = FrequencyAnalysisTool()
    noise_tool = NoiseAnalysisTool()

    results = {
        "frequency": {"correct": [], "wrong": [], "uncertain": []},
        "noise": {"correct": [], "wrong": [], "uncertain": []},
        "both_correct": [],
        "both_wrong": [],
        "freq_only": [],
        "noise_only": [],
        "both_uncertain": []
    }

    print("=" * 80)
    print("Tool Specialization Analysis")
    print("=" * 80)

    # AI 이미지 분석
    print("\n[AI Images Analysis]")
    for i, img_path in enumerate(ai_images):
        image = np.array(Image.open(img_path).convert("RGB"))

        freq_result = freq_tool.analyze(image)
        noise_result = noise_tool.analyze(image)

        freq_correct = freq_result.verdict == Verdict.AI_GENERATED
        noise_correct = noise_result.verdict == Verdict.AI_GENERATED or noise_result.verdict == Verdict.MANIPULATED

        freq_uncertain = freq_result.verdict == Verdict.UNCERTAIN
        noise_uncertain = noise_result.verdict == Verdict.UNCERTAIN

        item = {
            "filename": img_path.name,
            "freq_verdict": freq_result.verdict.value,
            "freq_score": freq_result.evidence.get("ai_generation_score", 0),
            "freq_confidence": freq_result.confidence,
            "noise_verdict": noise_result.verdict.value,
            "noise_score": noise_result.evidence.get("mvss_score", noise_result.evidence.get("ai_generation_score", 0)),
            "noise_confidence": noise_result.confidence,
        }

        # 분류
        if freq_correct and noise_correct:
            results["both_correct"].append(item)
        elif not freq_correct and not noise_correct and not freq_uncertain and not noise_uncertain:
            results["both_wrong"].append(item)
        elif freq_correct and not noise_correct and not noise_uncertain:
            results["freq_only"].append(item)
        elif noise_correct and not freq_correct and not freq_uncertain:
            results["noise_only"].append(item)
        elif freq_uncertain and noise_uncertain:
            results["both_uncertain"].append(item)

        if freq_correct:
            results["frequency"]["correct"].append(item)
        elif freq_uncertain:
            results["frequency"]["uncertain"].append(item)
        else:
            results["frequency"]["wrong"].append(item)

        if noise_correct:
            results["noise"]["correct"].append(item)
        elif noise_uncertain:
            results["noise"]["uncertain"].append(item)
        else:
            results["noise"]["wrong"].append(item)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/50 processed...")

    # 자연 이미지 분석
    print("\n[Natural Images Analysis]")
    for i, img_path in enumerate(nature_images):
        image = np.array(Image.open(img_path).convert("RGB"))

        freq_result = freq_tool.analyze(image)
        noise_result = noise_tool.analyze(image)

        freq_correct = freq_result.verdict == Verdict.AUTHENTIC
        noise_correct = noise_result.verdict == Verdict.AUTHENTIC

        freq_uncertain = freq_result.verdict == Verdict.UNCERTAIN
        noise_uncertain = noise_result.verdict == Verdict.UNCERTAIN

        item = {
            "filename": img_path.name,
            "freq_verdict": freq_result.verdict.value,
            "freq_score": freq_result.evidence.get("ai_generation_score", 0),
            "freq_confidence": freq_result.confidence,
            "noise_verdict": noise_result.verdict.value,
            "noise_score": noise_result.evidence.get("mvss_score", noise_result.evidence.get("ai_generation_score", 0)),
            "noise_confidence": noise_result.confidence,
        }

        # 분류
        if freq_correct and noise_correct:
            results["both_correct"].append(item)
        elif not freq_correct and not noise_correct and not freq_uncertain and not noise_uncertain:
            results["both_wrong"].append(item)
        elif freq_correct and not noise_correct and not noise_uncertain:
            results["freq_only"].append(item)
        elif noise_correct and not freq_correct and not freq_uncertain:
            results["noise_only"].append(item)
        elif freq_uncertain and noise_uncertain:
            results["both_uncertain"].append(item)

        if freq_correct:
            results["frequency"]["correct"].append(item)
        elif freq_uncertain:
            results["frequency"]["uncertain"].append(item)
        else:
            results["frequency"]["wrong"].append(item)

        if noise_correct:
            results["noise"]["correct"].append(item)
        elif noise_uncertain:
            results["noise"]["uncertain"].append(item)
        else:
            results["noise"]["wrong"].append(item)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/50 processed...")

    # 결과 출력
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    total = len(ai_images) + len(nature_images)

    print(f"\n[Individual Tool Performance]")
    print(f"Frequency Tool:")
    print(f"  Correct:   {len(results['frequency']['correct'])}/{total} ({len(results['frequency']['correct'])/total*100:.1f}%)")
    print(f"  Wrong:     {len(results['frequency']['wrong'])}/{total} ({len(results['frequency']['wrong'])/total*100:.1f}%)")
    print(f"  Uncertain: {len(results['frequency']['uncertain'])}/{total} ({len(results['frequency']['uncertain'])/total*100:.1f}%)")

    print(f"\nNoise Tool:")
    print(f"  Correct:   {len(results['noise']['correct'])}/{total} ({len(results['noise']['correct'])/total*100:.1f}%)")
    print(f"  Wrong:     {len(results['noise']['wrong'])}/{total} ({len(results['noise']['wrong'])/total*100:.1f}%)")
    print(f"  Uncertain: {len(results['noise']['uncertain'])}/{total} ({len(results['noise']['uncertain'])/total*100:.1f}%)")

    print(f"\n[Complementarity Analysis]")
    print(f"Both Correct:    {len(results['both_correct'])}/{total} ({len(results['both_correct'])/total*100:.1f}%) ← 두 툴 모두 정답")
    print(f"Both Wrong:      {len(results['both_wrong'])}/{total} ({len(results['both_wrong'])/total*100:.1f}%) ← 둘 다 오답 (문제!)")
    print(f"Freq Only:       {len(results['freq_only'])}/{total} ({len(results['freq_only'])/total*100:.1f}%) ← Frequency만 정답")
    print(f"Noise Only:      {len(results['noise_only'])}/{total} ({len(results['noise_only'])/total*100:.1f}%) ← Noise만 정답")
    print(f"Both Uncertain:  {len(results['both_uncertain'])}/{total} ({len(results['both_uncertain'])/total*100:.1f}%) ← 둘 다 불확실")

    # 상호보완성 계산
    covered = len(results['both_correct']) + len(results['freq_only']) + len(results['noise_only'])
    complementary_benefit = len(results['freq_only']) + len(results['noise_only'])

    print(f"\n[Complementarity Metrics]")
    print(f"Total Covered (Union): {covered}/{total} ({covered/total*100:.1f}%)")
    print(f"  - Both tools agree: {len(results['both_correct'])}")
    print(f"  - Complementary benefit: {complementary_benefit} (+{complementary_benefit/total*100:.1f}%)")
    print(f"")
    print(f"Uncovered (Gap): {total - covered}/{total} ({(total-covered)/total*100:.1f}%)")
    print(f"  - Both wrong: {len(results['both_wrong'])}")
    print(f"  - Both uncertain: {len(results['both_uncertain'])}")

    # 최악 케이스 분석
    print(f"\n[Worst Cases - Both Tools Wrong]")
    if results['both_wrong']:
        for item in results['both_wrong'][:5]:
            print(f"  {item['filename']}")
            print(f"    Freq: {item['freq_verdict']} (score={item['freq_score']:.2f})")
            print(f"    Noise: {item['noise_verdict']} (score={item['noise_score']:.2f})")
    else:
        print("  (None)")

    # JSON 저장
    output_path = Path("/root/Desktop/MAIFS/outputs/tool_specialization_analysis.json")
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n결과 저장: {output_path}")

    # 권장사항
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    freq_correct_rate = len(results['frequency']['correct']) / total
    noise_correct_rate = len(results['noise']['correct']) / total

    if freq_correct_rate < 0.8 or noise_correct_rate < 0.8:
        print("\n⚠️  문제: 각 툴의 정답률이 80% 미만")
        print(f"   - Frequency: {freq_correct_rate*100:.1f}%")
        print(f"   - Noise: {noise_correct_rate*100:.1f}%")
        print("\n해결 방안:")
        print("   1. Conservative threshold 설정 (확실할 때만 판정)")
        print("   2. UNCERTAIN 영역 확대")
        print("   3. 각 툴의 전문 영역 재정의")

    if len(results['both_wrong']) > total * 0.1:
        print(f"\n⚠️  문제: 두 툴 모두 오답인 케이스가 {len(results['both_wrong'])}개 ({len(results['both_wrong'])/total*100:.1f}%)")
        print("   → 이 케이스들은 다른 툴(Spatial, EXIF)로 보완 필요")

    if complementary_benefit < total * 0.2:
        print(f"\n⚠️  문제: 상호보완 효과가 {complementary_benefit/total*100:.1f}%로 낮음")
        print("   → 두 툴이 비슷한 이미지에 강점/약점 보임")
        print("   → 툴 간 차별화 필요")

if __name__ == "__main__":
    analyze_specialization()
