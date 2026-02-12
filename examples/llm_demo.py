#!/usr/bin/env python3
"""
MAIFS LLM Integration Demo
Claude API를 활용한 이미지 포렌식 분석 예시

Usage:
    # API 키 설정 후 실행
    export ANTHROPIC_API_KEY="your-api-key"
    python examples/llm_demo.py

    # 또는 코드에서 직접 설정
    python examples/llm_demo.py --api-key "your-api-key"
"""
import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


def demo_basic_analysis():
    """기본 분석 데모 (LLM 없이)"""
    print("\n" + "=" * 60)
    print("Demo 1: 기본 분석 (규칙 기반)")
    print("=" * 60)

    from src.agents.manager_agent import ManagerAgent

    # LLM 비활성화
    agent = ManagerAgent(use_llm=False)

    # 더미 이미지 생성
    image = np.random.rand(256, 256, 3).astype(np.float32)

    # 분석 실행
    report = agent.analyze(image)

    # 결과 출력
    print(f"\n최종 판정: {report.final_verdict.value}")
    print(f"신뢰도: {report.confidence:.1%}")
    print(f"처리 시간: {report.total_processing_time:.2f}초")

    # 사람이 읽기 쉬운 보고서
    human_report = agent.generate_human_report(report, language="ko")
    print("\n" + human_report)


def demo_llm_analysis(api_key: str = None):
    """LLM 기반 분석 데모"""
    print("\n" + "=" * 60)
    print("Demo 2: LLM 기반 분석 (Claude API)")
    print("=" * 60)

    from src.agents.manager_agent import ManagerAgent

    # LLM 활성화
    agent = ManagerAgent(use_llm=True, api_key=api_key)

    if not agent.use_llm:
        print("\n⚠️ Claude API를 사용할 수 없습니다.")
        print("   환경변수 ANTHROPIC_API_KEY를 설정하거나")
        print("   --api-key 옵션으로 API 키를 전달하세요.")
        print("\n   pip install anthropic")
        return

    # 더미 이미지 생성
    image = np.random.rand(256, 256, 3).astype(np.float32)

    # LLM 기반 분석
    report = agent.analyze_with_llm(image)

    # 결과 출력
    print(f"\n최종 판정: {report.final_verdict.value}")
    print(f"신뢰도: {report.confidence:.1%}")
    print(f"처리 시간: {report.total_processing_time:.2f}초")

    # 상세 추론
    print("\n[LLM 분석 결과]")
    print(report.detailed_reasoning)


def demo_real_image_analysis(image_path: str, api_key: str = None):
    """실제 이미지 분석 데모"""
    print("\n" + "=" * 60)
    print(f"Demo 3: 실제 이미지 분석")
    print(f"이미지: {image_path}")
    print("=" * 60)

    from src.agents.manager_agent import ManagerAgent

    # 이미지 로드
    if not os.path.exists(image_path):
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    image = np.array(img) / 255.0

    print(f"이미지 크기: {img.size}")

    # Manager Agent
    agent = ManagerAgent(use_llm=True, api_key=api_key)

    # 분석
    if agent.use_llm:
        report = agent.analyze_with_llm(image)
    else:
        report = agent.analyze(image)

    # 결과 출력
    print(f"\n최종 판정: {report.final_verdict.value}")
    print(f"신뢰도: {report.confidence:.1%}")
    print(f"처리 시간: {report.total_processing_time:.2f}초")

    # 전문가별 분석
    print("\n[전문가별 분석 결과]")
    for name, response in report.agent_responses.items():
        print(f"  - {name}: {response.verdict.value} ({response.confidence:.1%})")

    # 보고서
    human_report = agent.generate_human_report(report, language="ko")
    print("\n" + human_report)


def demo_claude_client():
    """Claude 클라이언트 직접 사용 데모"""
    print("\n" + "=" * 60)
    print("Demo 4: Claude 클라이언트 직접 사용")
    print("=" * 60)

    from src.llm.claude_client import ClaudeClient

    client = ClaudeClient()

    if not client.is_available:
        print("\n⚠️ Claude API를 사용할 수 없습니다.")
        print("   Fallback 모드로 동작합니다.")

    # 가상의 에이전트 응답
    mock_responses = {
        "frequency": {
            "verdict": "AI_GENERATED",
            "confidence": 0.85,
            "reasoning": "FFT 분석 결과 주기적인 아티팩트 패턴 발견. GAN 생성 이미지의 전형적인 특징."
        },
        "noise": {
            "verdict": "AI_GENERATED",
            "confidence": 0.75,
            "reasoning": "PRNU 분석 결과 카메라 센서 노이즈 패턴 부재. 자연 이미지와 다른 노이즈 분포."
        },
        "fatformer": {
            "verdict": "UNCERTAIN",
            "confidence": 0.50,
            "reasoning": "FatFormer 모델 미로드. 추가 분석 필요."
        },
        "spatial": {
            "verdict": "AI_GENERATED",
            "confidence": 0.80,
            "reasoning": "ViT 분석 결과 텍스처 일관성 부족. 지역적 불일치 발견."
        }
    }

    # 분석 요청
    response = client.analyze_forensics(mock_responses)

    print("\n[분석 결과]")
    print(response.content)


def main():
    parser = argparse.ArgumentParser(description="MAIFS LLM Demo")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument("--image", type=str, help="Image path to analyze")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["basic", "llm", "real", "client", "all"],
        default="all",
        help="Demo to run"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    print("=" * 60)
    print("        MAIFS LLM Integration Demo")
    print("=" * 60)

    if args.demo in ["basic", "all"]:
        demo_basic_analysis()

    if args.demo in ["llm", "all"]:
        demo_llm_analysis(api_key)

    if args.demo in ["client", "all"]:
        demo_claude_client()

    if args.demo == "real" or (args.demo == "all" and args.image):
        image_path = args.image or "HiNet-main/image/steg/00000.png"
        demo_real_image_analysis(image_path, api_key)

    print("\n" + "=" * 60)
    print("Demo 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
