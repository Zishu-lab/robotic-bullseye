#!/usr/bin/env python3
"""
靶心识别 Web 界面
启动: python app.py
然后在浏览器打开 http://localhost:5000
"""

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, Response, jsonify, request
import threading
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ========== HTML 模板 ==========
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>靶心识别系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #fff; }
        .wrap { max-width: 1100px; margin: 0 auto; padding: 15px; }
        h1 { text-align: center; padding: 15px; color: #00d4ff; }
        .box { background: #16213e; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .video { background: #000; border-radius: 8px; min-height: 350px; display: flex; align-items: center; justify-content: center; }
        .video img { max-width: 100%; border-radius: 8px; }
        .info { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.8); padding: 5px 10px; border-radius: 4px; font-size: 12px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 15px; }
        input { width: 100%; padding: 8px; border: 1px solid #333; border-radius: 4px; background: #0a0a1a; color: #fff; margin-bottom: 10px; }
        button { width: 100%; padding: 10px; border: none; border-radius: 6px; cursor: pointer; margin-bottom: 8px; font-size: 14px; }
        .btn1 { background: #00d4ff; color: #000; }
        .btn2 { background: #ff4757; color: #fff; }
        .stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; }
        .stat:last-child { border: none; }
        .val { color: #00d4ff; }
        .res { background: #0a0a1a; border-radius: 6px; padding: 10px; margin-top: 10px; border-left: 3px solid #00ff88; }
        .res .nm { color: #00ff88; font-weight: bold; }
        .res .cf { color: #888; font-size: 12px; }
        .empty { text-align: center; color: #555; padding: 15px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>靶心识别系统</h1>
        <div class="grid">
            <div>
                <div class="video" style="position:relative">
                    <img id="vid" src="/video">
                    <div class="info"><span id="stat">●</span> <span id="fps">-- FPS</span></div>
                </div>
            </div>
            <div class="box">
                <p style="margin-bottom:10px; color:#888">摄像头 URL</p>
                <input id="urlInput" value="http://10.24.100.139:8080/video">
                <button class="btn1" onclick="startDetect()">启动检测</button>
                <button class="btn2" onclick="stopDetect()">停止检测</button>
                <div class="stat"><span>状态:</span><span class="val" id="statusText">未启动</span></div>
                <div class="stat"><span>帧率:</span><span class="val" id="fpsText">-- FPS</span></div>
                <div class="stat"><span>检测数:</span><span class="val" id="countText">0</span></div>
                <h3 style="color:#00d4ff; margin: 10px 0;">识别结果</h3>
                <div id="results"><div class="empty">等待启动...</div></div>
            </div>
        </div>
    </div>
    <script>
        var updateInterval = null;

        function startDetect() {
            var url = document.getElementById('urlInput').value;
            fetch('/set_url', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: url})
            });
            fetch('/start', {method: 'POST'})
                .then(r => r.json())
                .then(d => {
                    alert(d.msg);
                    if (d.ok) {
                        document.getElementById('statusText').textContent = '运行中';
                        if (updateInterval) clearInterval(updateInterval);
                        updateInterval = setInterval(updateStatus, 300);
                    }
                });
        }

        function stopDetect() {
            fetch('/stop', {method: 'POST'})
                .then(r => r.json())
                .then(d => {
                    document.getElementById('statusText').textContent = '已停止';
                    if (updateInterval) {
                        clearInterval(updateInterval);
                        updateInterval = null;
                    }
                });
        }

        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('fpsText').textContent = d.fps + ' FPS';
                    document.getElementById('countText').textContent = d.results.length;
                    var html = '';
                    if (d.results.length > 0) {
                        for (var i = 0; i < d.results.length; i++) {
                            var r = d.results[i];
                            html += '<div class="res"><div class="nm">#' + (i+1) + ' ' + r.class + '</div>';
                            html += '<div class="cf">置信度: ' + (r.prob*100).toFixed(1) + '% | 检测: ' + (r.conf*100).toFixed(1) + '%</div></div>';
                        }
                    } else {
                        html = '<div class="empty">未检测到靶心</div>';
                    }
                    document.getElementById('results').innerHTML = html;
                });
        }
    </script>
</body>
</html>
'''

# ========== 后端逻辑 ==========
camera_url = 'http://10.24.100.139:8080/video'
running = False
detector = None
cap = None
frame = None
results = []
fps = 0
lock = threading.Lock()


class Detector:
    def __init__(self):
        self.yolo = None
        self.classifier = None
        self.classes = []
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self):
        root = Path(__file__).parent
        # YOLO
        from ultralytics import YOLO
        yolo_path = root / 'runs/detect/runs/detect/bullseye_train/weights/best.pt'
        if not yolo_path.exists():
            yolo_path = root / 'runs/detect/runs/detect/bullseye_optimized/weights/best.pt'
        logger.info(f"加载 YOLO 模型: {yolo_path}")
        self.yolo = YOLO(str(yolo_path))

        # Classifier - 优先使用新训练的 EfficientNet-B3 模型 (80.18%)
        cifar_path = root / 'experiments/runs/cifar100_optimized/best.pt'
        if not cifar_path.exists():
            cifar_path = root / 'models/classifier/final_classifier.pt'

        logger.info(f"加载分类器: {cifar_path}")
        ckpt = torch.load(cifar_path, map_location='cpu', weights_only=False)
        self.classes = ckpt.get('classes', [])

        # 检测模型类型
        state_dict = ckpt.get('model_state_dict', ckpt)
        is_efficientnet = any('classifier' in k or 'features.0.0' in k for k in state_dict.keys())

        if is_efficientnet:
            logger.info("加载 EfficientNet-B3 模型 (80.18%)")
            m = torchvision.models.efficientnet_b3(weights=None)
            m.features[0][0] = torch.nn.Conv2d(3, m.features[0][0].out_channels, 3, 1, 1, bias=False)
            m.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2, inplace=True),
                torch.nn.Linear(m.classifier[1].in_features, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.SiLU(inplace=True),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 100)
            )
        else:
            logger.info("加载 ResNet-50 模型")
            m = torchvision.models.resnet50(weights=None)
            m.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(m.fc.in_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, len(self.classes) if self.classes else 100)
            )

        # 加载权重
        try:
            m.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_state_dict = {k.replace('model.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            m.load_state_dict(new_state_dict, strict=False)

        m.to(self.device).eval()
        self.classifier = m
        logger.info(f"模型加载完成, classes={len(self.classes) if self.classes else 100}")

    def detect(self, frame):
        if not self.yolo:
            return []
        res = []
        for r in self.yolo(frame, verbose=False):
            for box in r.boxes:
                if box.conf[0] >= 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    t = self.transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.classifier(t)
                        p = torch.nn.functional.softmax(out, dim=1)
                        conf, idx = torch.max(p, 1)
                    class_name = self.classes[idx.item()] if idx.item() < len(self.classes) else 'unknown'
                    res.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': round(float(box.conf[0]), 2),
                        'class': class_name,
                        'prob': round(conf.item(), 3)
                    })
        return res


def loop():
    global running, cap, frame, results, fps
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        logger.error(f"无法连接摄像头: {camera_url}")
        running = False
        return
    logger.info("摄像头连接成功")
    t0 = time.time()
    fc = 0
    while running:
        ret, f = cap.read()
        if not ret:
            continue
        fc += 1
        res = detector.detect(f)
        vis = f.copy()
        for r in res:
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{r['class']} {r['prob']*100:.0f}%", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elapsed = time.time() - t0
        fps = fc / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        with lock:
            frame = vis
            results = res
    cap.release()
    logger.info("检测停止")


def gen():
    while True:
        with lock:
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.03)


@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')


@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start():
    global running, detector
    if running:
        return jsonify({'ok': False, 'msg': '已运行'})
    detector = Detector()
    detector.load()
    running = True
    threading.Thread(target=loop, daemon=True).start()
    return jsonify({'ok': True, 'msg': '已启动'})


@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    return jsonify({'ok': True, 'msg': '已停止'})


@app.route('/status')
def status():
    global running, fps, results
    return jsonify({'running': running, 'fps': round(fps, 1), 'results': results})


@app.route('/set_url', methods=['POST'])
def set_url():
    global camera_url
    camera_url = request.json.get('url', camera_url)
    logger.info(f"摄像头URL已更新: {camera_url}")
    return jsonify({'ok': True})


if __name__ == '__main__':
    print("\n靶心识别 Web 界面")
    print("=" * 40)
    print(f"  访问: http://localhost:5000")
    print(f"  摄像头: {camera_url}")
    print("=" * 40 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
