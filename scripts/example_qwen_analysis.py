#!/usr/bin/env python3
"""
MAIFS + Qwen vLLM 분석 예시

4개 전문가 에이전트가 Qwen을 사용하여 이미지를 분석하고
필요시 토론을 통해 합의에 도달합니다.

사전 요구사항:
1. vLLM 서버 실행: ./scripts/start_vllm_server.sh
2. 필요 패키지: pip install aiohttp

Usage:
    python scripts/example_qwen_analysis.py --image path/to/image.jpg
    python scripts/example_qwen_analysis.py --demo  # 데모 모드 (랜덤 이미지)
"""
import asyncio
import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


async def analyze_image_with_qwen(
    image_path: str = None,
    demo_mode: bool = False,
    vllm_url: str = "http://localhost:8000",
    enable_debate: bool = True
):
    """
    이미지 분석 수행

    Args:
        image_path: 분석할 이미지 경로
        demo_mode: 데모 모드 (랜덤 이미지 사용)
        vllm_url: vLLM 서버 URL
        enable_debate: 토론 활성화
    """
    from src.tools.frequency_tool import FrequencyAnalysisTool
    from src.tools.noise_tool import NoiseAnalysisTool
    from src.tools.watermark_tool import WatermarkTool
    from src.tools.spatial_tool import SpatialAnalysisTool
    from src.llm.qwen_maifs_adapter import QwenMAIFSAdapter

    print("=" * 60)
    print("MAIFS + Qwen vLLM 이미지 분석")
    print("=" * 60)

    # 1. 이미지 로드
    if demo_mode:
        print("\n[데모 모드] 랜덤 테스트 이미지 생성...")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    else:
        print(f"\n이미지 로드: {image_path}")
        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)

    print(f"이미지 크기: {image.shape}")

    # 2. Tool 분석 수행
    print("\n" + "-" * 60)
    print("Phase 1: Tool 분석 (4개 에이전트 병렬 실행)")
    print("-" * 60)

    tools = {
        "frequency": FrequencyAnalysisTool(),
        "noise": NoiseAnalysisTool(),
        "watermark": WatermarkTool(),
        "spatial": SpatialAnalysisTool()
    }

    tool_results = {}
    start_time = time.time()

    for name, tool in tools.items():
        print(f"  [{name}] 분석 중...")
        result = tool.analyze(image)
        tool_results[name] = result.evidence
        print(f"  [{name}] 완료 - {result.verdict.value} ({result.confidence:.1%})")

    tool_time = time.time() - start_time
    print(f"\nTool 분석 완료: {tool_time:.2f}초")

    # 3. Qwen LLM 해석
    print("\n" + "-" * 60)
    print("Phase 2: Qwen LLM 해석 (Batch Inference)")
    print("-" * 60)

    adapter = QwenMAIFSAdapter(
        base_url=vllm_url,
        enable_debate=enable_debate,
        max_debate_rounds=3
    )

    # 서버 상태 확인
    if not await adapter.client.health_check():
        print("ERROR: vLLM 서버에 연결할 수 없습니다.")
        print(f"서버 URL: {vllm_url}")
        print("./scripts/start_vllm_server.sh 를 먼저 실행하세요.")
        return

    print("vLLM 서버 연결 성공")

    # 배치 추론
    llm_start = time.time()
    analysis_results = await adapter.analyze_with_qwen(tool_results)
    llm_time = time.time() - llm_start

    print(f"\nLLM 해석 완료: {llm_time:.2f}초")

    # 결과 출력
    print("\n" + "-" * 60)
    print("에이전트별 분석 결과")
    print("-" * 60)

    for name, result in analysis_results.items():
        print(f"\n[{name.upper()}]")
        print(f"  판정: {result.verdict.value}")
        print(f"  신뢰도: {result.confidence:.1%}")
        print(f"  추론: {result.reasoning[:150]}...")
        if result.key_evidence:
            print(f"  핵심 증거: {result.key_evidence[:2]}")
        if result.raw_result:
            print(f"  지연시간: {result.raw_result.latency_ms:.0f}ms")

    # 4. 토론 (불일치 시)
    if enable_debate:
        print("\n" + "-" * 60)
        print("Phase 3: 토론 (불일치 시)")
        print("-" * 60)

        debate_start = time.time()
        debate_result = await adapter.conduct_debate(analysis_results)
        debate_time = time.time() - debate_start

        if debate_result.get("debate_conducted"):
            print(f"\n토론 수행: {debate_result['rounds']} 라운드")
            print(f"합의 도달: {'예' if debate_result['consensus_reached'] else '아니오'}")
            print(f"최종 판정: {debate_result['final_verdicts']}")
            print(f"토론 시간: {debate_time:.2f}초")

            # 토론 히스토리 출력
            if debate_result.get("history"):
                print("\n토론 내용:")
                for round_info in debate_result["history"]:
                    print(f"  라운드 {round_info['round']}:")
                    for exchange in round_info["exchanges"]:
                        print(f"    {exchange['challenger']} → {exchange['target']}")
                        print(f"    변경: {'예' if exchange['verdict_changed'] else '아니오'}")
        else:
            print(f"\n토론 불필요: {debate_result.get('reason', 'N/A')}")

    # 5. 최종 결과
    print("\n" + "=" * 60)
    print("최종 분석 결과")
    print("=" * 60)

    # 다수결 판정
    verdicts = [r.verdict.value for r in analysis_results.values()]
    from collections import Counter
    verdict_counts = Counter(verdicts)
    final_verdict = verdict_counts.most_common(1)[0][0]

    avg_confidence = sum(r.confidence for r in analysis_results.values()) / len(analysis_results)

    print(f"\n최종 판정: {final_verdict}")
    print(f"평균 신뢰도: {avg_confidence:.1%}")
    print(f"판정 분포: {dict(verdict_counts)}")
    print(f"\n총 소요 시간: {time.time() - start_time:.2f}초")
    print(f"  - Tool 분석: {tool_time:.2f}초")
    print(f"  - LLM 해석: {llm_time:.2f}초")

    await adapter.close()


def main():
    parser = argparse.ArgumentParser(description="MAIFS + Qwen 이미지 분석")
    parser.add_argument("--image", "-i", type=str, help="분석할 이미지 경로")
    parser.add_argument("--demo", action="store_true", help="데모 모드 (랜덤 이미지)")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="vLLM 서버 URL")
    parser.add_argument("--no-debate", action="store_true", help="토론 비활성화")

    args = parser.parse_args()

    if not args.demo and not args.image:
        parser.error("--image 또는 --demo 옵션이 필요합니다.")

    asyncio.run(analyze_image_with_qwen(
        image_path=args.image,
        demo_mode=args.demo,
        vllm_url=args.url,
        enable_debate=not args.no_debate
    ))


if __name__ == "__main__":
    main()
