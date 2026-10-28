# 위험 이벤트 평가 및 부분 실패 계측

## 이번 적용 결과

원본 453장과 실제 운영 DB 결과를 대조하여 331장의 정상 객체 분석 결과를 연결하였다. 나머지 122장은 대응하는 분석 로그가 없는 프레임으로 구분하였다. 이 122장을 곧바로 객체 미검출(FN)로 세지 않는다. 실제 이벤트의 FN은 검토된 이벤트 구간과 최종 경보의 매칭으로 판정한다.

| 항목 | 건수 | 의미 |
|---|---:|---|
| 보존 원본 | 453 | 박스·글자를 추가하기 전 촬영 입력 |
| 연결된 정상 객체 분석 | 331 | 카메라·파일명·촬영시각과 원본 해시를 확인한 입력 |
| 대응 분석 로그 없음 | 122 | 미처리·분석 대상 제외 등의 원인은 추가 확인 대상 |
| 구역 진입·격상 후보 | 20 | 사람 19건, 동물 1건 |
| 독립 쓰러짐 시작 후보 | 0 | `fallEvent=ENTER` |
| 실제 저장된 최종 위험 경보 | 0 | `riskAlarms` 기록. 구역 후보와 별도 집계 |

기준 자료는 `result/kpi/event-review/validated-bundle/`이다. `summary.json`의 전체 성능 상태는 `metrics_ready=false`이다. 전체 구간의 독립 검토 GT가 없으므로 Precision·Recall·F1·FPR·감지 정확도를 자동으로 채우지 않았다.

## 원본과 실제 결과 연결

`prepare_event_evaluation.py`는 원본 보존 manifest와 읽기 전용 DB 스냅샷을 입력받는다. 원본 파일의 크기·SHA-256, 폴더의 카메라 ID, 파일명의 촬영시각과 DB 촬영시각을 검사한다. 동일 프레임에 대응하는 DB 행이 여러 개이면 임의 선택하지 않고 중단한다.

```bash
python prepare_event_evaluation.py \
  result/kpi/candidate-investigation/raw-capture \
  --results result/kpi/candidate-investigation/raw-capture/production-results.json \
  --output result/kpi/new-event-review
```

| 출력 파일 | 내용 |
|---|---|
| `frames.csv` | 원본 경로·해시·분석 로그 ID·전체 상태·객체 분석 상태·부분 오류 |
| `zone_candidates.csv` | 정상 객체 분석에서 발생한 구역 `ENTER/ESCALATE` |
| `fall_candidates.csv` | 정상 객체 분석에서 발생한 독립 쓰러짐 `ENTER` |
| `issued_alerts.csv` | 실제 저장된 최종 경보. `time`은 연결된 촬영시각, `issued_at`은 저장된 경보 발생시각 |
| `gt.template.csv` | 검토 후 작성할 이벤트 GT의 빈 양식 |
| `coverage.template.csv` | 위험·비위험을 모두 검토한 구간의 빈 양식 |
| `summary.json` | 연결 건수·카메라별 통계·자료 해시·평가 준비 상태 |

DB는 KST 벽시계를 `TIMESTAMP WITHOUT TIME ZONE`으로 저장한다. 이 스냅샷의 Prisma `Z` 표시는 저장 표현이며, 도구는 UTC→KST 변환을 추가하지 않는다. 다른 DB에서 출력한 UTC 자료를 그대로 입력하지 않는다.

빈 양식은 정답이나 검토 완료 구간이 아니다. 촬영 범위 전체를 자동으로 비위험 구간으로 바꾸지 않는다. 경보의 기기 발성·표출시각도 생성하지 않는다.

## 후보 평가와 최종 경보 평가

후보 CSV는 `--candidates`, 최종 경보 CSV는 `--alerts`로 입력한다. 두 옵션은 동시에 사용할 수 없다. JSON의 `settings.evaluation_scope`에 후보 평가인지 시스템 경보 평가인지 기록한다.

```bash
python evaluate_kpi.py \
  --candidates result/kpi/new-event-review/zone_candidates.csv \
  --gt reviewed-zone-gt.csv --coverage reviewed-coverage.csv --classes person

python evaluate_kpi.py \
  --alerts result/kpi/new-event-review/issued_alerts.csv \
  --gt reviewed-risk-gt.csv --coverage reviewed-coverage.csv --classes person
```

구역 점유와 트램 접근 조건을 충족한 실제 위험은 평가 대상을 구분한다. 최종 경보가 0건이라는 이유만으로 구역 진입 후보를 최종 경보 대신 사용하지 않는다.

## 내부 영상 검토 결과

카메라 4의 원본 57장을 확대 순서도와 전체 구역 순서도로 검토하였다. 설정된 위험 폴리곤 안에 사람이 관측되는 두 구역 점유 구간을 내부 GT 초안으로 작성하였다. 여러 사람이 함께 있는 구간은 하나의 카메라 점유 사건으로 묶었으며, 생물학적 개인 ID나 개별 객체 수의 정답을 지정하지 않았다.

| 구간 | 최초 내부 관측 | 마지막 내부 관측 |
|---|---|---|
| 1 | 13:49:35.812 | 13:49:46.656 |
| 2 | 13:49:48.284 | 13:49:54.392 |

카메라 4의 검토 범위는 13:49:30.104~13:49:59.744이다. 촬영·경계 관측의 불확실성을 반영해 매칭 허용 오차를 0.5초로 설정하였다. 후보 평가에서 TP 2건, FP 0건, FN 0건, 같은 GT 구간의 추가 후보 2건을 확인하였다. 이 추가 후보는 다른 사람에 대한 검출일 수 있으므로 자동 억제하지 않았다.

이는 AI의 내부 영상 검토에 따른 **두 사건의 구성요소 진단**이다. 독립된 사람의 GT 검수가 아니며, 최종 트램 위험 경보 평가·전체 사람 인식률·1차년도 목표 달성의 근거로 사용할 수 없다. 평가값은 `internal-camera4/evaluation.json`, 검토 범위·한계는 같은 폴더의 `provenance.json`에 보존하였다.

전체 후보 20건의 클래스도 확인하였다. 사람으로 볼 수 있는 후보 14건, 동물 1건을 확인하였고 카메라 8의 작은 원거리 후보 5건은 `UNKNOWN`으로 남겼다. 이 검토는 전체 GT가 아니므로 15건을 분모로 정밀도를 계산하지 않는다. 자료는 `candidate-class-review.csv`이다.

## 부분 실패 구분

통합 분석에서 객체·얼굴 처리에 실패하더라도 성공한 군중 계수는 기존 계약대로 전달한다. 실패한 구성요소는 `raw.component_errors`와 기존 `errorMessage` 필드에 기록한다. 오류 발생 이전에 생성된 미완성 객체·얼굴 결과는 전송하지 않고 해당 처리시간도 성공 계측에서 제외한다.

평가 준비 도구는 전체 `COMPLETED`만으로 객체 분석 성공을 판단하지 않는다. `component_errors.objects`가 있으면 객체 처리를 `FAILED`로 표시하고 후보를 생성하지 않는다. 오류 표시도 객체 처리시간도 없는 이전 결과는 `UNKNOWN`으로 표시한다. 따라서 부분 실패나 구성요소 비활성화를 유효한 빈 객체 검출로 간주하지 않는다.

이 변경은 실패 원인을 구분하는 계측 보완이다. 객체를 관측하지 못한 동안 실제 위험이 없었다는 보장을 추가하지 않는다. 검출 모델·임계값·트램 접근 정책은 유지한다.

## 검증 및 근거

- 관련 분석·평가 회귀 테스트 154개 통과.
- 원본 453장의 해시와 연결된 DB 파일명·촬영시각 검증.
- 실제 부분 오류의 callback 보존, 미완성 결과 제거, 실패 표본 제외를 회귀 테스트로 확인.
- GT·검토 구간은 자동 생성하지 않고 빈 양식으로 제공.

상세 근거는 `result/kpi/event-review/verification.txt`, `validated-bundle/summary.json`, `candidate-class-review-summary.json`, `internal-camera4/provenance.json`이다. 서비스 반영 후 상태는 같은 폴더의 `deployment-health.json`과 `deployment-log.txt`에 기록한다.
