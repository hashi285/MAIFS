"""
LLM Integration Tests
Claude API 통합 테스트
"""
import pytest
import numpy as np
import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestClaudeClient:
    """Claude 클라이언트 테스트"""

    def test_import_claude_client(self):
        """Claude 클라이언트 import 테스트"""
        from src.llm.claude_client import ClaudeClient, LLMResponse
        assert ClaudeClient is not None
        assert LLMResponse is not None

    def test_client_initialization_without_api_key(self):
        """API 키 없이 클라이언트 초기화"""
        from src.llm.claude_client import ClaudeClient

        # 환경변수 임시 제거
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            client = ClaudeClient(api_key=None)
            assert client is not None
            # API 키 없으면 사용 불가
            assert not client.is_available
        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_fallback_analysis(self):
        """Fallback 분석 테스트 (API 없이)"""
        from src.llm.claude_client import ClaudeClient

        client = ClaudeClient(api_key=None)

        # 가상의 에이전트 응답
        mock_responses = {
            "frequency": {
                "verdict": "AI_GENERATED",
                "confidence": 0.85,
                "reasoning": "주파수 분석 결과 GAN 패턴 발견"
            },
            "noise": {
                "verdict": "AI_GENERATED",
                "confidence": 0.75,
                "reasoning": "PRNU 패턴 불일치"
            },
            "watermark": {
                "verdict": "AUTHENTIC",
                "confidence": 0.60,
                "reasoning": "워터마크 미발견"
            },
            "spatial": {
                "verdict": "AI_GENERATED",
                "confidence": 0.80,
                "reasoning": "공간 분석 이상 탐지"
            }
        }

        response = client.analyze_forensics(mock_responses)

        assert response is not None
        assert response.model == "rule-based-fallback"
        assert "verdict" in response.content

    def test_llm_response_structure(self):
        """LLMResponse 구조 테스트"""
        from src.llm.claude_client import LLMResponse

        response = LLMResponse(
            content='{"verdict": "AUTHENTIC", "confidence": 0.9}',
            model="test-model",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn"
        )

        assert response.content is not None
        assert response.model == "test-model"
        assert response.usage["input_tokens"] == 100

        # to_dict 테스트
        d = response.to_dict()
        assert d["model"] == "test-model"
        assert d["stop_reason"] == "end_turn"

    def test_fallback_report_generation(self):
        """Fallback 보고서 생성 테스트"""
        from src.llm.claude_client import ClaudeClient

        client = ClaudeClient(api_key=None)

        mock_responses = {
            "frequency": {
                "verdict": "AI_GENERATED",
                "confidence": 0.85,
            },
            "noise": {
                "verdict": "AI_GENERATED",
                "confidence": 0.75,
            }
        }

        # 한국어 보고서
        report_ko = client.generate_report(
            verdict="AI_GENERATED",
            confidence=0.80,
            agent_responses=mock_responses,
            language="ko"
        )

        assert "MAIFS" in report_ko
        assert "AI_GENERATED" in report_ko or "AI 생성" in report_ko

        # 영어 보고서
        report_en = client.generate_report(
            verdict="AI_GENERATED",
            confidence=0.80,
            agent_responses=mock_responses,
            language="en"
        )

        assert "MAIFS" in report_en
        assert "AI_GENERATED" in report_en


class TestManagerAgentLLMIntegration:
    """Manager Agent LLM 통합 테스트"""

    def test_manager_agent_with_llm_disabled(self):
        """LLM 비활성화 상태에서 Manager Agent 테스트"""
        from src.agents.manager_agent import ManagerAgent

        agent = ManagerAgent(use_llm=False)

        assert agent is not None
        assert not agent.use_llm
        assert agent.llm_client is None

    def test_manager_agent_llm_initialization(self):
        """Manager Agent LLM 초기화 테스트"""
        from src.agents.manager_agent import ManagerAgent

        # API 키 없이 초기화
        agent = ManagerAgent(use_llm=True, api_key=None)

        assert agent is not None
        # API 키 없으면 LLM 비활성화됨
        # (anthropic 패키지 미설치 시에도 비활성화)

    def test_analyze_with_llm_fallback(self):
        """LLM fallback 분석 테스트"""
        from src.agents.manager_agent import ManagerAgent

        agent = ManagerAgent(use_llm=False)

        # 더미 이미지
        image = np.random.rand(256, 256, 3).astype(np.float32)

        # 분석 실행
        report = agent.analyze(image)

        assert report is not None
        assert report.final_verdict is not None
        assert 0.0 <= report.confidence <= 1.0
        assert len(report.agent_responses) > 0

    def test_generate_human_report(self):
        """사람이 읽기 쉬운 보고서 생성 테스트"""
        from src.agents.manager_agent import ManagerAgent

        agent = ManagerAgent(use_llm=False)

        # 더미 이미지
        image = np.random.rand(256, 256, 3).astype(np.float32)

        # 분석 실행
        report = agent.analyze(image)

        # 보고서 생성
        human_report = agent.generate_human_report(report, language="ko")

        assert human_report is not None
        assert "MAIFS" in human_report
        assert "판정" in human_report or "Verdict" in human_report

        # 영어 보고서
        human_report_en = agent.generate_human_report(report, language="en")
        assert "MAIFS" in human_report_en

    def test_analyze_with_llm_method(self):
        """analyze_with_llm 메서드 테스트"""
        from src.agents.manager_agent import ManagerAgent

        agent = ManagerAgent(use_llm=False)

        # 더미 이미지
        image = np.random.rand(256, 256, 3).astype(np.float32)

        # LLM 분석 (fallback 모드)
        report = agent.analyze_with_llm(image)

        assert report is not None
        assert report.final_verdict is not None
        assert report.summary is not None


class TestLLMWithRealImage:
    """실제 이미지로 LLM 테스트"""

    @pytest.fixture
    def sample_image(self):
        """샘플 이미지 로드"""
        from PIL import Image
        import os

        # 실제 이미지 경로 (png 확장자)
        image_paths = [
            "/root/Desktop/MAIFS/HiNet-main/image/steg/00000.png",
            "/root/Desktop/MAIFS/TruFor-main/test_docker/images/tampered1.png",
        ]

        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    return np.array(img) / 255.0
                except Exception:
                    continue

        # 더미 이미지
        return np.random.rand(512, 512, 3).astype(np.float32)

    def test_full_analysis_pipeline(self, sample_image):
        """전체 분석 파이프라인 테스트"""
        from src.agents.manager_agent import ManagerAgent

        agent = ManagerAgent(use_llm=False)
        report = agent.analyze(sample_image)

        assert report is not None
        assert report.final_verdict is not None

        # 보고서 출력
        human_report = agent.generate_human_report(report, language="ko")
        print("\n" + human_report)


class TestLLMIntegrationWithAPI:
    """API가 있을 때 LLM 통합 테스트"""

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_real_claude_analysis(self):
        """실제 Claude API 테스트"""
        from src.llm.claude_client import ClaudeClient

        client = ClaudeClient()

        if not client.is_available:
            pytest.skip("Claude API not available")

        mock_responses = {
            "frequency": {
                "verdict": "AI_GENERATED",
                "confidence": 0.85,
                "reasoning": "주파수 분석 결과 GAN 패턴 발견"
            },
            "noise": {
                "verdict": "AI_GENERATED",
                "confidence": 0.75,
                "reasoning": "PRNU 패턴 불일치"
            }
        }

        response = client.analyze_forensics(mock_responses)

        assert response is not None
        assert response.model != "rule-based-fallback"
        print(f"\nClaude Response:\n{response.content}")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_manager_with_real_api(self):
        """실제 API로 Manager Agent 테스트"""
        from src.agents.manager_agent import ManagerAgent

        agent = ManagerAgent(use_llm=True)

        if not agent.use_llm:
            pytest.skip("LLM not available")

        image = np.random.rand(256, 256, 3).astype(np.float32)
        report = agent.analyze_with_llm(image)

        assert report is not None
        print(f"\nLLM Report:\n{report.detailed_reasoning}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
