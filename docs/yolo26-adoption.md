# YOLO26 적용 기록

정지 자전거의 누락·오분류 개선을 위해 정확도 중심 헤드와 고해상도 ONNX를 생성하고, 분석 서버의 모델별 입력 처리 및 선택적 TensorRT FP16 실행 경로를 추가하였다. 운영 모델 교체와 서버 재시작은 수행하지 않았다.

## 기능별 상태

| 기능 | 적용 상태 | 검증·선행 조건 |
|---|---|---|
| DFL-free 구조 | 기존 YOLO26 구조에 포함 | 실행 설정으로 별도 추가하지 않는다. |
| NMS-free one-to-one 헤드 | 기존 지원 유지 | `(1, 300, 6)` 출력을 처리한다. |
| 정확도 중심 one-to-many 헤드 | ONNX 생성·런타임 처리 구현 | 클래스별 NMS를 수행한다. 각 데이터에서 정확도 개선을 검증한다. |
| NMS 내장 ONNX | 출력 형식 자동 구분 구현 | `end2end=False`라도 마지막 출력 차원이 6이면 처리된 검출 결과로 해석한다. detect 작업만 허용한다. |
| 640·1280·직사각형 입력 | 모델별 자동 처리 구현 | 1280 주 모델과 640 보조 모델의 혼합 실행을 지원한다. 고정 입력은 모델에서 확인한다. |
| 동적 입력 | 기존 `size`를 공간 크기 대체값으로 사용 | 기본 대체값은 640이다. 여러 배치의 동시 처리는 지원하지 않는다. |
| FP16 ONNX 입력 | dtype 처리·좌표 복원 테스트 완료 | 실제 GPU 정밀도·성능 검증은 필요하다. CPU에서 FP16 export를 요청하면 중단한다. |
| TensorRT FP16 | 선택 실행·초기화 검증 구현 | 기본 비활성화. GPU·TensorRT 라이브러리가 있는 환경에서 실측이 필요하다. |
| YOLO26s·m·l·x | 같은 detect ONNX 로딩 경로 사용 | 가중치 다운로드가 DNS 오류로 실패하여 실제 검증은 미실시하였다. |
| Progressive Loss·STAL·MuSGD | 학습 단계 기능 | 이번 작업에서는 재학습하지 않았다. 추론 설정만으로 추가할 수 없다. |
| P2·P6 구조 | 미적용 | 공식 문서에 따르면 학습된 가중치가 제공되지 않는 아키텍처 변형이다. 라벨 데이터와 학습·검증이 필요하다. |
| 세그멘테이션·포즈·깊이·OBB | 현재 2D 객체 탐지 범위에서 제외 | 기존 합의에 따른 bbox 기반 쓰러짐 판정을 유지한다. |
| INT8 | 미적용 | 보정 데이터와 별도 정확도·GPU 검증이 필요하다. |

공식 문서의 여러 작업은 서로 다른 가중치와 출력을 사용한다. 현재 서버의 detect 입력에 pose·segment 등의 모델이 지정되면 오류로 처리한다. [YOLO26 공식 문서](https://docs.ultralytics.com/ko/models/yolo26), [엔드투엔드 탐지](https://docs.ultralytics.com/ko/guides/end2end-detection)를 기준으로 범위를 확인하였다.

## ONNX 생성

모델 생성 도구는 학습 체크포인트를 읽는 개발 환경에서 실행한다. 서비스 런타임에 Ultralytics 패키지를 추가하지 않는다. 검증 환경은 Ultralytics 8.4.60, ONNX 1.23.2이다.

```bash
python export_yolo26.py yolo26n-objv1-150.pt \
  result/models/objects365_yolo26n_accuracy_1280.onnx \
  --head accuracy --imgsz 1280

python export_yolo26.py yolo26n-objv1-150.pt \
  result/models/objects365_yolo26n_end2end_1280.onnx \
  --head end2end --imgsz 1280
```

입력 체크포인트는 로컬 파일이어야 하며, 기존 출력 파일은 덮어쓰지 않는다. `--half`는 CUDA가 있는 생성 환경에서만 사용한다. 생성 도구는 export 후 `end2end` 메타데이터를 검사하여 요청한 헤드가 실제로 생성되었는지 확인한다.

Ultralytics 8.4.60은 `end2end=False/True`와 `nms=False`로 헤드를 구분한다. 현재 공식 문서는 `nms=None/False`를 제시하므로, 도구는 설치 버전의 설정 항목을 확인하여 해당 인자를 선택한다. export 시 `simplify=False`, `dynamic=False`, `opset=17`을 사용한다. ONNX 그래프 단순화 도구를 자동 설치하지 않는다.

이 작업에서 생성한 파일은 `result/yolo26-adoption/objects365-yolo26n-accuracy-640.onnx`, `objects365-yolo26n-accuracy-1280.onnx`, `objects365-yolo26n-end2end-1280.onnx`이다. `result/`는 Git 관리 대상이 아니므로 배포 시 가중치 파일을 별도로 전달한다.

## 분석 서버 설정

생성한 주 모델을 컨테이너의 `/app/result/models/`에 전달하거나 읽기 전용으로 마운트한 뒤 `OBJECT_MODEL_PATH`에 해당 경로를 지정한다. 기존 보조 모델은 `MOBILITY_MODEL_PATH`로 지정한다. 1280 모델은 추가적인 메모리·연산량을 요구하므로 대상 GPU에서 검증한 후 운영 경로를 변경한다.

```bash
OBJECT_MODEL_PATH=/app/result/models/objects365_yolo26n_accuracy_1280.onnx
MOBILITY_MODEL_PATH=/app/result/models/mobility_yolov8s.onnx
OBJECT_TENSORRT=0
```

TensorRT 실행은 ONNX Runtime의 TensorRT provider와 호환 라이브러리를 설치한 환경에서 `OBJECT_TENSORRT=1`로 설정한다. 통합 서버는 `--object-tensorrt`, 객체 전용 서버는 `--tensorrt`도 지원한다. provider 순서는 TensorRT FP16, CUDA, CPU이며, TensorRT를 요청하였지만 등록·초기화가 실패하면 모델 로딩을 실패 처리한다. provider 등록 여부만으로 모델 전체가 TensorRT에서 처리되거나 처리 속도가 개선되었다고 판단하지 않는다. [ONNX Runtime TensorRT 문서](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)를 참조한다.

TensorRT 라이브러리는 이번 변경에서 Docker 이미지에 설치하지 않았다. 서버의 CUDA·TensorRT·ONNX Runtime 호환성과 메모리, 초기 로딩 시간, 전체 분석 지연 시간을 별도로 검증한다. 얼굴·군중 모델의 실행 경로는 변경하지 않았다.

## 이미지 검증

고정 위치의 같은 객체가 계속 보이는 사진에 다음 도구를 사용한다. `--reference`는 원본 이미지의 비교 대상 박스이며, IoU 0.3 이상인 검출 결과 중 최고 신뢰도 class를 집계한다. 전체 장면의 정답 라벨을 대체하지 않는다.

```bash
python validate_yolo26.py '/path/to/images' \
  result/models/objects365_yolo26n_accuracy_1280.onnx \
  result/yolo26-validation \
  --reference 998 260 1055 347 \
  --mobility-model result/models/mobility_yolov8s.onnx \
  --device cpu
```

검증 도구는 새 출력 디렉터리에 `detections.jsonl`과 `summary.json`을 기록한다. CUDA 검증은 `--device cuda`로 지정한다. 처리 시간은 파일 읽기와 HTTP·callback·대기열을 제외한 검출기 처리 시간이며, 실시간 전체 처리 성능으로 해석하지 않는다.

197장 데이터는 동일한 정지 자전거와 기존 검출 박스·글자가 포함된 이미지이다. 이를 원본 카메라 프레임의 검출률, 다른 카메라의 정확도 또는 전체 객체 성능으로 일반화하지 않는다. 실행 결과는 `result/yolo26-adoption/validation-summary.json`과 각 profile의 `detections.jsonl`에 기록한다.

### 197장 검증 결과

| 주 모델·처리 구성 | `bicycle` | `person` 오분류 | 누락 |
|---|---:|---:|---:|
| 이전 640 NMS-free + 보조 모델·보정 | 81 | 9 | 107 |
| 640 one-to-many 단독 | 0 | 15 | 182 |
| 1280 NMS-free 단독 | 46 | 0 | 151 |
| 1280 one-to-many 단독 | 79 | 0 | 118 |
| 1280 one-to-many + 640 보조 모델·보정 | 115 | 0 | 82 |

위 표는 프레임별 검출을 독립적으로 수행한 기준 결과이다. 이후 추가한 이전 검출 위치 기반 재검출·선택적 영상 보정 결과는 [자전거 누락 개선 기록](bicycle-miss-upgrade.md)을 참조한다. 다른 객체의 전체 정답 라벨이 없으므로 사람·동물·이동수단 전체의 정확도 유지 여부는 추가 검증이 필요하다. 이 결과만으로 운영 모델을 자동 교체하지 않는다.

ONNX 생성과 검증 도구의 실제 실행을 확인하였다. 단위·회귀 테스트 95개가 통과하였으며, HTTP 연결 테스트 1개는 실행 환경의 소켓 제한으로 제외하였다. TensorRT 선택·초기화 오류 처리는 대체 세션으로 검증하였고, 실제 TensorRT·FP16 GPU 처리는 미검증 상태이다.

## 추가 학습 조건

자전거·사람·동물·이동수단의 깨끗한 원본 프레임과 전체 객체 라벨을 확보하고, 카메라·촬영 구간 단위로 학습·검증 데이터를 분리한다. 현재 197장은 같은 장면의 연속 이미지이며 기존 오분류 박스가 포함되어 있어, 이를 그대로 정답 라벨로 사용하거나 무작위 분할하여 일반화 성능으로 보고하지 않는다. 부분 라벨만으로 학습하면 다른 사람·이동수단이 배경으로 취급될 수 있으므로 라벨 검토가 선행되어야 한다.

새 모델의 적용 기준은 자전거 누락·오분류 개선, 사람·동물·이동수단 검출 유지, 대상 GPU에서의 전체 분석 지연 시간 검증이다. 현재의 `person`·`animal` 정규화, `alertEvent`·`fallEvent` 응답 및 위험구역·트램 정책은 이번 모델 설정 변경에서 유지한다.
