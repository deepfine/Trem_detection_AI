FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-serve.txt

COPY anomaly_rules.py camera_zones.py detect_anomalies.py crowd_analysis_protocol.py crowd_analysis_server.py crowd_counter.py frame_faces.py frame_objects.py object_analysis_server.py zone_annotator.py ./

ENV PYTHONUNBUFFERED=1 \
    ANALYSIS_HOST=0.0.0.0 \
    ANALYSIS_PORT=8080 \
    FRAME_ANALYZED_DIR=/upload/visit_servant/analyzed \
    CROWD_MODEL_PATH=/app/result/models/dm_count_qnrf.pth \
    CROWD_DEVICE=cuda:0 \
    VISIT_SERVANT_RESULT_URL=http://api:3535/crowd/results \
    OBJECT_MODEL_PATH=/app/result/models/objects365_yolo26n.onnx \
    MOBILITY_MODEL_PATH=/app/result/models/mobility_yolov8s.onnx \
    OBJECT_DEVICE_ID=0 \
    OBJECT_RESULT_URL=http://api:3535/object/results \
    FACE_MODEL_PATH=/app/result/models/scrfd_det_10g.onnx \
    FACE_DEVICE_ID=0 \
    FACE_DET_SIZE=960x544 \
    FACE_THRESHOLD=0.35

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD python3 -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:' + os.environ.get('ANALYSIS_PORT', '8080') + '/health', timeout=2).read()"

CMD ["python3", "crowd_analysis_server.py"]
