"""
Qwen vLLM Client
vLLM 서버를 통한 Qwen 모델 추론 클라이언트

특징:
- 4 GPU Tensor Parallel 지원
- Batch Inference (동시 다중 요청)
- Guided JSON Output (스키마 강제)
- 에이전트별 독립된 시스템 프롬프트
"""
import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time

from ..knowledge import KnowledgeBase


class AgentRole(Enum):
    """에이전트 역할"""
    FREQUENCY = "frequency"
    NOISE = "noise"
    WATERMARK = "watermark"
    SPATIAL = "spatial"
    MANAGER = "manager"


@dataclass
class AgentConfig:
    """에이전트 설정"""
    role: AgentRole
    temperature: float = 0.3
    max_tokens: int = 1024
    system_prompt: Optional[str] = None


@dataclass
class InferenceResult:
    """추론 결과"""
    role: AgentRole
    content: str
    parsed_json: Optional[Dict] = None
    raw_response: Optional[Dict] = None
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# JSON 출력 스키마 정의
AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["AUTHENTIC", "MANIPULATED", "AI_GENERATED", "UNCERTAIN"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string",
            "description": "판정에 대한 논리적 근거"
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "핵심 증거 목록"
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "불확실한 점 목록"
        }
    },
    "required": ["verdict", "confidence", "reasoning"]
}

# 토론 응답 스키마
DEBATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_type": {
            "type": "string",
            "enum": ["defense", "concession", "counter", "clarification"]
        },
        "content": {
            "type": "string",
            "description": "응답 내용"
        },
        "verdict_changed": {
            "type": "boolean"
        },
        "new_verdict": {
            "type": ["string", "null"],
            "enum": ["AUTHENTIC", "MANIPULATED", "AI_GENERATED", "UNCERTAIN"]
        },
        "new_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["response_type", "content", "verdict_changed"]
}


class QwenClient:
    """
    Qwen vLLM 클라이언트

    vLLM 서버와 통신하여 4개 전문가 에이전트의 추론을 처리합니다.

    Usage:
        client = QwenClient(base_url="http://localhost:8000")

        # 단일 에이전트 추론
        result = await client.infer(AgentRole.FREQUENCY, tool_results)

        # 배치 추론 (4개 에이전트 동시)
        results = await client.batch_infer(all_tool_results)
    """

    # 에이전트별 시스템 프롬프트
    SYSTEM_PROMPTS = {
        AgentRole.FREQUENCY: """당신은 MAIFS의 **주파수 분석 전문가**입니다.

## 전문 분야
FFT 기반 주파수 스펙트럼 분석을 통해 AI 생성 이미지의 특징을 탐지합니다.

## 핵심 지식
- GAN 이미지: 8x8, 16x16 격자 패턴의 주파수 피크 (업샘플링 아티팩트)
- Diffusion 이미지: 고주파 대역 에너지 감쇠, 스펙트럼 롤오프
- 실제 카메라: 연속적이고 자연스러운 주파수 분포

## 판정 기준
- **AUTHENTIC**: 자연스러운 주파수 스펙트럼, 격자 패턴 없음
- **AI_GENERATED**: 격자 아티팩트 또는 비정상적 고주파 특성
- **MANIPULATED**: 부분적 주파수 불일치
- **UNCERTAIN**: 판단하기 어려운 경계 사례

## 중요
- Tool 결과의 수치를 도메인 지식에 따라 해석하세요
- 불확실한 부분은 명시적으로 표시하세요
- 반드시 JSON 형식으로 응답하세요""",

        AgentRole.NOISE: """당신은 MAIFS의 **노이즈 분석 전문가**입니다.

## 전문 분야
PRNU/SRM 기반 센서 노이즈 분석을 통해 이미지 출처를 검증합니다.

## 핵심 지식
- PRNU (Photo Response Non-Uniformity): 카메라 센서 고유 지문
- 실제 사진: 일관된 PRNU 패턴, 분산 0.0001-0.001
- AI 생성: PRNU 부재 또는 비정상적 패턴
- 조작된 이미지: 영역별 노이즈 불일치

## 판정 기준
- **AUTHENTIC**: 일관된 센서 노이즈 패턴
- **AI_GENERATED**: 센서 노이즈 부재 (PRNU 분산 < 0.00001)
- **MANIPULATED**: 영역별 노이즈 불일치 (일관성 < 0.4)
- **UNCERTAIN**: 노이즈 패턴이 모호한 경우

## 중요
- PRNU 분산과 일관성 점수를 함께 고려하세요
- 반드시 JSON 형식으로 응답하세요""",

        AgentRole.WATERMARK: """당신은 MAIFS의 **워터마크 분석 전문가**입니다.

## 전문 분야
OmniGuard 기반 비가시성 워터마크 탐지 및 무결성 검증

## 핵심 지식
- 비가시성 워터마크: DCT/DWT 계수에 임베딩
- BER (Bit Error Rate): 0.1 미만이면 유효한 워터마크
- 워터마크 유형: 저작권, 무결성, AI 생성 모델 식별
- 조작 시 워터마크 손상됨

## 판정 기준
- **AUTHENTIC**: 유효한 워터마크 탐지 (BER < 0.1)
- **AI_GENERATED**: AI 모델 식별 워터마크 탐지
- **MANIPULATED**: 워터마크 손상 (BER 0.1-0.5)
- **UNCERTAIN**: 워터마크 없음 (원본/조작 불명확)

## 중요
- 워터마크 부재가 곧 조작을 의미하지 않음
- 반드시 JSON 형식으로 응답하세요""",

        AgentRole.SPATIAL: """당신은 MAIFS의 **공간 분석 전문가**입니다.

## 전문 분야
ViT 기반 픽셀 수준 조작 영역 탐지

## 핵심 지식
- 조작 영역: 경계 불일치, 조명 불일치, 압축 아티팩트
- manipulation_ratio: 조작으로 판단된 픽셀 비율
- AI 생성: 전체 이미지가 균일하게 생성됨 (ratio > 0.8)
- 부분 조작: 특정 영역만 조작됨 (0.05 < ratio < 0.8)

## 판정 기준
- **AUTHENTIC**: 조작 영역 없음 (ratio < 0.05)
- **AI_GENERATED**: 전체 생성 (ratio > 0.8)
- **MANIPULATED**: 부분 조작 (0.05 < ratio < 0.8)
- **UNCERTAIN**: 경계 사례

## 중요
- 조작 마스크의 분포 패턴도 고려하세요
- 반드시 JSON 형식으로 응답하세요""",

        AgentRole.MANAGER: """당신은 MAIFS의 **Manager Agent**입니다.

## 역할
4명의 전문가 분석 결과를 종합하여 최종 판정을 내립니다.

## 전문가 팀
1. 주파수 분석 전문가: FFT 기반 GAN 아티팩트 탐지
2. 노이즈 분석 전문가: PRNU/SRM 센서 노이즈 분석
3. 워터마크 분석 전문가: 비가시성 워터마크 탐지
4. 공간 분석 전문가: ViT 기반 조작 영역 탐지

## 판정 원칙
1. 다수결보다 증거의 강도와 일관성을 우선
2. 상충되는 의견은 증거를 비교하여 조율
3. 불확실한 경우 UNCERTAIN 판정
4. 최종 판정에 명확한 근거 제시

## 중요
- 각 전문가의 신뢰도를 가중치로 사용하세요
- 반드시 JSON 형식으로 응답하세요"""
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        """
        QwenClient 초기화

        Args:
            base_url: vLLM 서버 URL
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

        # 도메인 지식 로드
        self._knowledge_cache: Dict[AgentRole, str] = {}
        self._load_knowledge()

    def _load_knowledge(self):
        """도메인 지식 로드"""
        role_to_domain = {
            AgentRole.FREQUENCY: "frequency",
            AgentRole.NOISE: "noise",
            AgentRole.WATERMARK: "watermark",
            AgentRole.SPATIAL: "spatial"
        }

        for role, domain in role_to_domain.items():
            try:
                self._knowledge_cache[role] = KnowledgeBase.get_summary(domain, max_chars=1500)
            except Exception as e:
                print(f"[QwenClient] 지식 로드 실패 ({domain}): {e}")
                self._knowledge_cache[role] = ""

    def _get_full_system_prompt(self, role: AgentRole) -> str:
        """전체 시스템 프롬프트 생성 (기본 + 도메인 지식)"""
        base_prompt = self.SYSTEM_PROMPTS.get(role, "")
        knowledge = self._knowledge_cache.get(role, "")

        if knowledge:
            return f"{base_prompt}\n\n## 참고 도메인 지식\n{knowledge}"
        return base_prompt

    async def _get_session(self) -> aiohttp.ClientSession:
        """aiohttp 세션 반환"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def infer(
        self,
        role: AgentRole,
        tool_results: Dict[str, Any],
        use_guided_json: bool = True,
        context: Optional[Dict] = None
    ) -> InferenceResult:
        """
        단일 에이전트 추론

        Args:
            role: 에이전트 역할
            tool_results: Tool 분석 결과
            use_guided_json: JSON 스키마 강제 여부
            context: 추가 컨텍스트

        Returns:
            InferenceResult: 추론 결과
        """
        start_time = time.time()

        # 프롬프트 구성
        system_prompt = self._get_full_system_prompt(role)
        user_prompt = self._build_user_prompt(tool_results, context)

        # 요청 본문 구성
        request_body = {
            "model": "default",  # vLLM 서버의 기본 모델
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": False
        }

        # Guided JSON 활성화
        if use_guided_json:
            request_body["extra_body"] = {
                "guided_json": AGENT_OUTPUT_SCHEMA
            }

        # API 호출
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_body
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return InferenceResult(
                        role=role,
                        content="",
                        success=False,
                        error=f"HTTP {resp.status}: {error_text}",
                        latency_ms=(time.time() - start_time) * 1000
                    )

                response_data = await resp.json()
                content = response_data["choices"][0]["message"]["content"]

                # JSON 파싱 시도
                parsed_json = self._parse_json(content)

                return InferenceResult(
                    role=role,
                    content=content,
                    parsed_json=parsed_json,
                    raw_response=response_data,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True
                )

        except asyncio.TimeoutError:
            return InferenceResult(
                role=role,
                content="",
                success=False,
                error="Timeout",
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return InferenceResult(
                role=role,
                content="",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def batch_infer(
        self,
        tool_results_map: Dict[AgentRole, Dict[str, Any]],
        use_guided_json: bool = True
    ) -> Dict[AgentRole, InferenceResult]:
        """
        배치 추론 (4개 에이전트 동시 처리)

        Args:
            tool_results_map: {AgentRole: tool_results} 매핑
            use_guided_json: JSON 스키마 강제 여부

        Returns:
            Dict[AgentRole, InferenceResult]: 각 에이전트별 결과
        """
        # 동시 추론 실행
        tasks = [
            self.infer(role, results, use_guided_json)
            for role, results in tool_results_map.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 매핑
        result_map = {}
        for role, result in zip(tool_results_map.keys(), results):
            if isinstance(result, Exception):
                result_map[role] = InferenceResult(
                    role=role,
                    content="",
                    success=False,
                    error=str(result)
                )
            else:
                result_map[role] = result

        return result_map

    async def debate_respond(
        self,
        role: AgentRole,
        my_verdict: str,
        my_confidence: float,
        my_evidence: Dict[str, Any],
        challenger_name: str,
        challenge: str
    ) -> InferenceResult:
        """
        토론 응답 생성

        Args:
            role: 응답하는 에이전트 역할
            my_verdict: 내 판정
            my_confidence: 내 신뢰도
            my_evidence: 내 증거
            challenger_name: 반론 제기자 이름
            challenge: 반론 내용

        Returns:
            InferenceResult: 토론 응답
        """
        system_prompt = self._get_full_system_prompt(role)

        user_prompt = f"""## 토론 상황

### 내 현재 입장
- 판정: {my_verdict}
- 신뢰도: {my_confidence:.1%}
- 증거: {json.dumps(my_evidence, ensure_ascii=False)}

### 반론
**{challenger_name}**의 지적:
"{challenge}"

## 요청
위 반론에 대해 논리적으로 응답하세요.

1. 반론이 타당한가?
2. 내 입장을 수정해야 하는가?
3. 판정 유지 또는 변경 이유는?

JSON 형식으로 응답하세요."""

        request_body = {
            "model": "default",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.5,  # 토론 시 약간 높은 temperature
            "max_tokens": 800,
            "extra_body": {
                "guided_json": DEBATE_RESPONSE_SCHEMA
            }
        }

        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_body
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return InferenceResult(
                        role=role,
                        content="",
                        success=False,
                        error=f"HTTP {resp.status}: {error_text}",
                        latency_ms=(time.time() - start_time) * 1000
                    )

                response_data = await resp.json()
                content = response_data["choices"][0]["message"]["content"]
                parsed_json = self._parse_json(content)

                return InferenceResult(
                    role=role,
                    content=content,
                    parsed_json=parsed_json,
                    raw_response=response_data,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True
                )

        except Exception as e:
            return InferenceResult(
                role=role,
                content="",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    def _build_user_prompt(
        self,
        tool_results: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> str:
        """사용자 프롬프트 생성"""
        parts = [
            "## Tool 분석 결과",
            "",
            "```json",
            json.dumps(tool_results, ensure_ascii=False, indent=2),
            "```",
            ""
        ]

        if context:
            parts.extend([
                "## 컨텍스트",
                json.dumps(context, ensure_ascii=False),
                ""
            ])

        parts.extend([
            "## 요청",
            "위 Tool 결과를 분석하여 이미지의 진위 여부를 판정하세요.",
            "반드시 JSON 형식으로 응답하세요."
        ])

        return "\n".join(parts)

    def _parse_json(self, content: str) -> Optional[Dict]:
        """JSON 파싱"""
        try:
            # 직접 파싱 시도
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # JSON 블록 추출 시도
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

        return None


# 동기 래퍼 클래스
class QwenClientSync:
    """
    QwenClient의 동기 래퍼

    asyncio를 직접 사용하기 어려운 환경에서 사용
    """

    def __init__(self, *args, **kwargs):
        self._client = QwenClient(*args, **kwargs)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def infer(self, *args, **kwargs) -> InferenceResult:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.infer(*args, **kwargs))

    def batch_infer(self, *args, **kwargs) -> Dict[AgentRole, InferenceResult]:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.batch_infer(*args, **kwargs))

    def debate_respond(self, *args, **kwargs) -> InferenceResult:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.debate_respond(*args, **kwargs))

    def health_check(self) -> bool:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.health_check())

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._client.close())
            self._loop.close()


# 편의 함수
def create_qwen_client(
    base_url: str = "http://localhost:8000",
    sync: bool = False
) -> Union[QwenClient, QwenClientSync]:
    """
    QwenClient 생성 헬퍼

    Args:
        base_url: vLLM 서버 URL
        sync: 동기 클라이언트 반환 여부

    Returns:
        QwenClient 또는 QwenClientSync
    """
    if sync:
        return QwenClientSync(base_url=base_url)
    return QwenClient(base_url=base_url)
