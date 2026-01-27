"""
EXIF 메타데이터 분석 도구
카메라 정보, 촬영 설정, GPS 등 메타데이터를 통한 진위 판정
"""
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from .base_tool import BaseTool, ToolResult, Verdict


class ExifAnalysisTool(BaseTool):
    """
    EXIF 메타데이터 분석 도구

    EXIF 데이터 분석을 통해:
    - 실제 카메라 제조사/모델 확인
    - 촬영 설정 (ISO, 조리개, 셔터 속도) 검증
    - GPS 위치 정보 확인
    - 소프트웨어 편집 흔적 탐지
    - AI 생성 도구 메타데이터 탐지
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(
            name="exif_analyzer",
            description="EXIF 메타데이터 분석. "
                       "카메라 정보와 촬영 설정을 확인하여 실제 촬영 여부를 판단합니다.",
            device=device
        )
        self._is_loaded = True  # 외부 모델 불필요

    def load_model(self) -> None:
        """모델 로드 (EXIF 분석은 외부 모델 불필요)"""
        self._is_loaded = True

    def _extract_exif(self, image_path: str) -> Dict[str, Any]:
        """
        EXIF 데이터 추출

        Args:
            image_path: 이미지 파일 경로

        Returns:
            EXIF 데이터 딕셔너리
        """
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()

            if exif_data is None:
                return {}

            # EXIF 태그를 읽기 쉬운 이름으로 변환
            decoded = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)

                # GPS 데이터 처리
                if tag == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id in value:
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_value = value[gps_tag_id]
                        # IFDRational을 float로 변환
                        if hasattr(gps_value, 'numerator') and hasattr(gps_value, 'denominator'):
                            gps_value = float(gps_value)
                        gps_data[gps_tag] = gps_value
                    decoded[tag] = gps_data
                else:
                    # IFDRational 객체를 float로 변환
                    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                        value = float(value)
                    # 바이트 데이터는 문자열로 변환
                    elif isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    # 튜플이면 리스트로 변환 (JSON 호환)
                    elif isinstance(value, tuple):
                        value = [float(v) if hasattr(v, 'numerator') else v for v in value]
                    decoded[tag] = value

            return decoded

        except Exception as e:
            return {"error": str(e)}

    def _analyze_camera_info(self, exif: Dict[str, Any]) -> Dict[str, Any]:
        """
        카메라 정보 분석

        실제 카메라: 제조사, 모델, 렌즈 정보 존재
        AI 생성: 메타데이터 없음 또는 AI 도구 시그니처
        """
        camera_make = exif.get("Make", "").strip()
        camera_model = exif.get("Model", "").strip()
        lens_model = exif.get("LensModel", "").strip()
        software = exif.get("Software", "").strip()

        # 알려진 카메라 제조사
        known_makes = [
            "Canon", "Nikon", "Sony", "Fujifilm", "Olympus", "Panasonic",
            "Leica", "Pentax", "Samsung", "Apple", "Google", "Huawei",
            "OnePlus", "Xiaomi", "OPPO", "vivo", "Hasselblad", "Phase One"
        ]

        # AI 생성 도구 시그니처
        ai_signatures = [
            "Midjourney", "DALL-E", "Stable Diffusion", "Adobe Firefly",
            "Leonardo.ai", "Playground", "RunwayML", "Artbreeder",
            "DeepAI", "NightCafe", "Craiyon", "DreamStudio"
        ]

        has_camera_make = any(make.lower() in camera_make.lower() for make in known_makes)
        has_ai_signature = any(sig.lower() in software.lower() for sig in ai_signatures)

        # 점수 계산
        authenticity_score = 0.0

        if has_camera_make:
            authenticity_score += 0.4
        if camera_model:
            authenticity_score += 0.3
        if lens_model:
            authenticity_score += 0.2
        if software and not has_ai_signature:
            authenticity_score += 0.1

        return {
            "camera_make": camera_make,
            "camera_model": camera_model,
            "lens_model": lens_model,
            "software": software,
            "has_camera_info": has_camera_make,
            "has_ai_signature": has_ai_signature,
            "authenticity_score": min(authenticity_score, 1.0)
        }

    def _analyze_shooting_params(self, exif: Dict[str, Any]) -> Dict[str, Any]:
        """
        촬영 파라미터 분석

        ISO, 조리개, 셔터 속도 등이 실제 카메라 설정과 일치하는지 확인
        """
        iso = exif.get("ISOSpeedRatings")
        aperture = exif.get("FNumber")
        shutter_speed = exif.get("ExposureTime")
        focal_length = exif.get("FocalLength")
        exposure_mode = exif.get("ExposureMode")
        white_balance = exif.get("WhiteBalance")

        # 파라미터 존재 점수
        param_count = sum([
            iso is not None,
            aperture is not None,
            shutter_speed is not None,
            focal_length is not None,
            exposure_mode is not None,
            white_balance is not None
        ])

        # 6개 중 몇 개 존재하는지
        param_score = param_count / 6.0

        # 값의 합리성 검증
        plausible = True
        issues = []

        if iso is not None:
            if isinstance(iso, int):
                # ISO는 보통 100-51200 범위
                if iso < 50 or iso > 102400:
                    plausible = False
                    issues.append(f"비정상적 ISO: {iso}")

        if aperture is not None:
            # 조리개는 보통 f/1.0 - f/32
            if isinstance(aperture, (int, float)):
                if aperture < 1.0 or aperture > 64:
                    plausible = False
                    issues.append(f"비정상적 조리개: f/{aperture}")

        return {
            "iso": iso,
            "aperture": aperture,
            "shutter_speed": shutter_speed,
            "focal_length": focal_length,
            "param_completeness": param_score,
            "values_plausible": plausible,
            "issues": issues
        }

    def _analyze_metadata_consistency(self, exif: Dict[str, Any]) -> Dict[str, Any]:
        """
        메타데이터 일관성 분석

        날짜/시간, 파일 수정 이력, 소프트웨어 정보가 일관성 있는지 확인
        """
        datetime_original = exif.get("DateTimeOriginal")
        datetime_digitized = exif.get("DateTimeDigitized")
        datetime_modified = exif.get("DateTime")

        # 날짜 정보 존재 여부
        has_timestamps = any([datetime_original, datetime_digitized, datetime_modified])

        # 소프트웨어 편집 흔적
        software = exif.get("Software", "").lower()
        editing_software = ["photoshop", "lightroom", "gimp", "affinity", "pixelmator", "snapseed"]
        has_editing_trace = any(sw in software for sw in editing_software)

        return {
            "datetime_original": datetime_original,
            "datetime_digitized": datetime_digitized,
            "datetime_modified": datetime_modified,
            "has_timestamps": has_timestamps,
            "has_editing_trace": has_editing_trace,
            "software_detected": software if software else None
        }

    def analyze(self, image: np.ndarray, image_path: Optional[str] = None) -> ToolResult:
        """
        EXIF 메타데이터 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3) - 사용되지 않음 (호환성 유지)
            image_path: 이미지 파일 경로 (EXIF 읽기에 필수)

        Returns:
            ToolResult: EXIF 분석 결과
        """
        start_time = time.time()

        if image_path is None:
            # 경로가 없으면 분석 불가
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={"error": "image_path required for EXIF analysis"},
                explanation="EXIF 분석을 위해서는 이미지 파일 경로가 필요합니다.",
                processing_time=time.time() - start_time
            )

        try:
            # EXIF 추출
            exif_data = self._extract_exif(image_path)

            if not exif_data or "error" in exif_data:
                # EXIF 없음 → AI 생성 가능성
                return ToolResult(
                    tool_name=self.name,
                    verdict=Verdict.AI_GENERATED,
                    confidence=0.7,
                    evidence={
                        "exif_present": False,
                        "reason": "No EXIF data found"
                    },
                    explanation=(
                        "EXIF 메타데이터가 없습니다. "
                        "AI 생성 이미지는 카메라 메타데이터가 없는 경우가 많습니다. "
                        "또는 의도적으로 제거되었을 수 있습니다."
                    ),
                    processing_time=time.time() - start_time
                )

            # 분석 수행
            camera_analysis = self._analyze_camera_info(exif_data)
            param_analysis = self._analyze_shooting_params(exif_data)
            consistency_analysis = self._analyze_metadata_consistency(exif_data)

            # 종합 점수
            camera_score = camera_analysis["authenticity_score"]
            param_score = param_analysis["param_completeness"]

            # AI 시그니처가 있으면 즉시 AI_GENERATED
            if camera_analysis["has_ai_signature"]:
                verdict = Verdict.AI_GENERATED
                confidence = 0.95
                explanation = (
                    f"AI 생성 도구 시그니처가 감지되었습니다: {camera_analysis['software']}. "
                    f"이 이미지는 AI가 생성한 것으로 확인됩니다."
                )

            # 실제 카메라 정보가 충분히 있으면 AUTHENTIC
            elif camera_score > 0.7 and param_score > 0.5:
                verdict = Verdict.AUTHENTIC
                confidence = (camera_score + param_score) / 2
                explanation = (
                    f"실제 카메라 메타데이터가 확인되었습니다. "
                    f"제조사: {camera_analysis['camera_make']}, "
                    f"모델: {camera_analysis['camera_model']}. "
                    f"촬영 파라미터가 정상적으로 기록되어 있습니다."
                )

            # 편집 흔적이 있으면 MANIPULATED 가능성
            elif consistency_analysis["has_editing_trace"]:
                verdict = Verdict.MANIPULATED
                confidence = 0.6
                explanation = (
                    f"이미지 편집 소프트웨어 흔적이 감지되었습니다: "
                    f"{consistency_analysis['software_detected']}. "
                    f"원본 이미지가 편집되었을 가능성이 있습니다."
                )

            # 카메라 정보는 있지만 불완전
            elif camera_score > 0.3:
                verdict = Verdict.UNCERTAIN
                confidence = 0.5
                explanation = (
                    f"일부 카메라 정보가 확인되었으나 불완전합니다. "
                    f"카메라 점수: {camera_score:.2%}, 파라미터 점수: {param_score:.2%}"
                )

            # EXIF는 있지만 카메라 정보 없음 → 의심
            else:
                verdict = Verdict.UNCERTAIN
                confidence = 0.4
                explanation = (
                    "EXIF 데이터는 존재하지만 카메라 정보가 부족합니다. "
                    "일부 편집 도구나 스크린샷은 EXIF를 생성하지만 카메라 정보는 포함하지 않습니다."
                )

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "exif_present": True,
                    "camera_analysis": camera_analysis,
                    "shooting_params": param_analysis,
                    "consistency": consistency_analysis,
                    "raw_exif_sample": {
                        k: v for k, v in list(exif_data.items())[:10]
                    }
                },
                explanation=explanation,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={"error": str(e)},
                explanation=f"EXIF 분석 중 오류 발생: {str(e)}",
                processing_time=processing_time
            )
