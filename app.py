#!/usr/bin/env python3
"""
靶心识别 Web 界面
使用 DetectionService 服务模块
"""

import cv2
import sys
from pathlib import Path
from flask import Flask, Response, jsonify, request
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.services.detection_service import get_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 获取服务单例
service = get_service()

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


def gen():
    """视频流生成器"""
    while True:
        frame = service.get_current_frame()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')


@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start():
    if service.is_running:
        return jsonify({'ok': False, 'msg': '已运行'})
    success = service.start_stream()
    return jsonify({
        'ok': success,
        'msg': '已启动' if success else '启动失败'
    })


@app.route('/stop', methods=['POST'])
def stop():
    service.stop_stream()
    return jsonify({'ok': True, 'msg': '已停止'})


@app.route('/status')
def status():
    return jsonify(service.get_status())


@app.route('/set_url', methods=['POST'])
def set_url():
    url = request.json.get('url', '')
    logger.info(f"摄像头URL更新: {url}")
    return jsonify({'ok': True})


if __name__ == '__main__':
    print("\n靶心识别 Web 界面")
    print("=" * 40)
    print(f"  访问: http://localhost:5000")
    print(f"  配置文件: config/model_config.yaml")
    print("=" * 40 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
