# 本地骨骼追踪服务（更稳的进阶对比）

这个项目新增了一个本地 Python 服务，用于在 **进阶对比模式** 下做更稳定的多人场景主体追踪与骨骼标注关键帧输出（避免骨骼跑到别人身上）。

## 1) 安装依赖

在 `/Users/enosh/Desktop/快卷吧妈妈` 目录：

```bash
pip3 install --upgrade --no-cache-dir mediapipe opencv-python fastapi uvicorn python-multipart numpy
```

> 说明：这是 Demo 形态安装，会与当前系统里其它 Python 包产生版本冲突风险（例如 numpy）。建议后续改成虚拟环境（venv/conda）。

## 2) 启动服务

```bash
uvicorn backend_server:app --host 127.0.0.1 --port 8787
```

健康检查：

```bash
curl http://127.0.0.1:8787/health
```

## 3) 前端如何使用

打开 `/Users/enosh/Desktop/快卷吧妈妈/kuaijuan-mama-v2.html`（推荐用本地 HTTP server 而非 file://）。

当你在页面选择 **进阶对比模式** 并点击 “开始 AI 分析” 时：
- 前端会优先调用 `http://localhost:8787/api/compare/keyframes`
- 后端返回带骨骼标注的关键帧（base64 jpg）
- 前端把这些关键帧画回页面 canvas，后续流程保持不变

如果本地服务不可用，前端会回退到浏览器端 MediaPipe（稳定性较差）。

