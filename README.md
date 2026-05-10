# 快卷吧妈妈｜AI 舞蹈夸夸教练（Demo）

一个单页 HTML 小应用：上传跳舞视频 → 生成温柔的「夸夸 + 小建议」报告 → 导出分享卡。

> 注意：这是黑客松/演示形态。请勿上传敏感或隐私视频。API Key 由你在页面手动填写；正式产品应改为后端代理，避免在前端暴露 Key。

---

## 目录结构

- `kuaijuan-mama-v2.html`：前端（UI + 逻辑都在一个文件里）
- `vendor/mediapipe/...`：本地 MediaPipe 资源（避免 CDN/GCS 不可用）
- `backend_server.py`：本地后端（更稳的关键帧/骨骼/指标计算）
- `pose_detector.py`：MediaPipe Pose 检测封装
- `metrics.py`：量化指标计算（角度/幅度/流畅度/稳定等）
- `README-local-backend.md`：后端安装/启动说明（更细）

---

## 快速开始（推荐）

### 1) 启动前端（不要用 file:// 打开）

在项目目录运行：

```bash
python3 -m http.server 8000
```

浏览器打开：

```text
http://127.0.0.1:8000/kuaijuan-mama-v2.html
```

### 2)（可选但强烈推荐）启动本地后端

本地后端用于 **进阶对比模式**：
- 更稳地输出骨骼标注关键帧 / 对比小视频
- 计算并输出量化指标（`metrics.py`），用于喂给 Gemini/DeepSeek 生成更具体建议

安装依赖（Demo 方式）：

```bash
pip3 install --upgrade --no-cache-dir mediapipe opencv-python fastapi uvicorn python-multipart numpy
```

启动：

```bash
uvicorn backend_server:app --host 127.0.0.1 --port 8787
```

健康检查：

```bash
curl http://127.0.0.1:8787/health
```

---

## 使用流程

### A. 纯夸夸（只上传我的视频）

1. 选择「纯夸夸」
2. 上传「我的视频」
3. 选择时间段
4. 提取关键帧
5. 填 API Key（建议用 Gemini）
6. 生成报告并导出分享卡

### B. 进阶对比（导师 vs 我）

1. 选择「进阶对比」
2. 上传「参考视频」+「我的视频」
3. 选择两段视频的时间段
4. 选择主体（锁定导师与自己，避免骨骼跑到别人身上）
5. 提取关键帧
6. 填 API Key（Gemini/DeepSeek 均可）
7. 生成报告并导出分享卡

> 若本地后端已启动，页面会优先使用 `http://127.0.0.1:8787`；否则会回退到浏览器端 MediaPipe（稳定性较差）。

---

## 常见问题

### 1) 为什么一定要用 `http://` 打开？

`file://` 属于不安全的 origin，浏览器会阻止 WASM/模块加载，导致 MediaPipe 加载失败或行为不一致。请使用 `python3 -m http.server`。

### 2) 关键帧/骨骼识别不稳定怎么办？

- 尽量选：全身入镜、光线更好、遮挡更少、画面更清晰的片段
- 进阶对比模式建议启动本地后端（稳定性明显提升）

---

## 评委/分享者（最小交付清单）

建议仓库至少包含：
- `kuaijuan-mama-v2.html`
- `backend_server.py`, `pose_detector.py`, `metrics.py`
- `vendor/`（避免外网不可用导致无法运行）
- `README.md`

