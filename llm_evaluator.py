"""
LLM Evaluator - 舞蹈动作教练点评生成器

Takes structured dance metrics from metrics.py and sends them to
Gemini or DeepSeek API to generate a narrative coaching report in Chinese,
with professional analysis and encouragement.

Usage:
    from llm_evaluator import generate_coaching_report
    report = generate_coaching_report(api_key, "gemini", metrics)
    print(report)
"""

from typing import Optional


# ---------------------------------------------------------------------------
# 1. Prompt Builder
# ---------------------------------------------------------------------------

def _quality_label(score: float) -> str:
    """Map a numeric score to a Chinese quality label."""
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "有待提高"
    else:
        return "需要重点练习"


def _format_per_joint(joint_data: dict) -> str:
    """Format per-joint angle data into readable Chinese lines."""
    lines = []
    joint_names_cn = {
        "left_elbow": "左肘",
        "right_elbow": "右肘",
        "left_shoulder": "左肩",
        "right_shoulder": "右肩",
        "left_knee": "左膝",
        "right_knee": "右膝",
        "left_hip": "左髋",
        "right_hip": "右髋",
        "left_ankle": "左踝",
        "right_ankle": "右踝",
        "left_wrist": "左腕",
        "right_wrist": "右腕",
    }
    for joint_name, info in joint_data.items():
        cn = joint_names_cn.get(joint_name, joint_name)
        similarity = info.get("similarity", 0)
        label = _quality_label(similarity * 100)
        lines.append(
            f"  - {cn}：老师平均角度 {info.get('teacher_mean', '?'):.1f}°，"
            f"学生平均角度 {info.get('student_mean', '?'):.1f}°，"
            f"相似度 {similarity * 100:.1f}%（{label}）"
        )
    return "\n".join(lines)


def _format_amplitude(amplitude_data: dict) -> str:
    """Format amplitude data into readable Chinese lines."""
    lines = []
    point_names_cn = {
        "left_wrist": "左腕",
        "right_wrist": "右腕",
        "left_ankle": "左踝",
        "right_ankle": "右踝",
    }
    per_point = amplitude_data.get("per_point", {})
    for point_name, info in per_point.items():
        cn = point_names_cn.get(point_name, point_name)
        score = info.get("score", 0)
        label = _quality_label(score)
        lines.append(
            f"  - {cn}：老师幅度 {info.get('t_amp', 0):.2f}，"
            f"学生幅度 {info.get('s_amp', 0):.2f}，"
            f"幅度比 {info.get('ratio', 0):.2f}，得分 {score:.1f}（{label}）"
        )
    return "\n".join(lines)


def build_coaching_prompt(metrics: dict) -> str:
    """
    Build a structured Chinese prompt for the LLM.

    The prompt includes metric explanations, actual data with quality labels,
    and detailed output formatting instructions for the coaching report.
    """
    overall = metrics.get("overall_score", 0)
    overall_label = _quality_label(overall)

    breakdown = metrics.get("breakdown", {})
    joint_angles = metrics.get("joint_angles", {})
    amplitude = metrics.get("amplitude", {})
    smoothness = metrics.get("smoothness", {})
    stiffness = metrics.get("stiffness", {})

    # Dimension scores with labels
    dim_scores = {
        "关节角度相似度": breakdown.get("joint_angle_similarity", 0),
        "动作幅度比": breakdown.get("amplitude_ratio", 0),
        "动作流畅度": breakdown.get("motion_smoothness", 0),
        "下半身稳定度": breakdown.get("lower_body_stiffness", 0),
    }

    # Build dimension summary
    dim_lines = []
    for dim_name, dim_score in dim_scores.items():
        label = _quality_label(dim_score)
        dim_lines.append(f"  - {dim_name}：{dim_score:.1f} 分（{label}）")
    dim_summary = "\n".join(dim_lines)

    # Format detailed data
    joint_text = ""
    per_joint = joint_angles.get("per_joint", {})
    if per_joint:
        joint_text = _format_per_joint(per_joint)

    amplitude_text = ""
    if amplitude.get("per_point"):
        amplitude_text = _format_amplitude(amplitude)

    smoothness_text = ""
    if smoothness:
        t_s = smoothness.get("teacher_smoothness", 0)
        s_s = smoothness.get("student_smoothness", 0)
        s_score = smoothness.get("smoothness_score", 0)
        smoothness_text = (
            f"  老师流畅度 {t_s:.2f}，学生流畅度 {s_s:.2f}，"
            f"流畅度得分 {s_score:.1f}（{_quality_label(s_score)}）"
        )

    stiffness_text = ""
    if stiffness:
        t_st = stiffness.get("teacher_stiffness", 0)
        s_st = stiffness.get("student_stiffness", 0)
        st_score = stiffness.get("stiffness_score", 0)
        stiffness_text = (
            f"  老师摇晃度 {t_st:.2f}，学生摇晃度 {s_st:.2f}，"
            f"稳定度得分 {st_score:.1f}（{_quality_label(st_score)}）"
        )

    prompt = f"""你是一位拥有20年经验的专业舞蹈教练，正在分析一位学生的舞蹈动作。
请根据下面的量化指标数据，生成一份专业、具体、充满鼓励的舞蹈动作教练点评报告。

## 数据说明
- **关节角度相似度**：比较学生和老师各关节（肘、肩、膝、髋等）弯曲角度的相似程度。分数越高说明学生模仿得越到位。
- **动作幅度比**：比较四肢末端（手腕、脚踝）运动幅度的匹配程度。低于100%说明动作偏小，高于100%说明动作偏大。
- **动作流畅度**：衡量动作轨迹的平滑程度，越平滑分数越高。
- **下半身稳定度**：衡量下半身是否存在不自主的晃动，分数越高说明下盘越稳（低于90分说明比老师晃动更多）。

## 学生数据

### 总体得分：{overall:.1f} 分（{overall_label}）

### 各维度得分
{dim_summary}

### 关节角度详情
{joint_text if joint_text else "（无详细数据）"}

### 动作幅度详情
{amplitude_text if amplitude_text else "（无详细数据）"}

### 流畅度
{smoothness_text if smoothness_text else "（无详细数据）"}

### 稳定度
{stiffness_text if stiffness_text else "（无详细数据）"}

## 输出格式要求
请严格按照以下四个部分输出报告：

### 1. 整体表现
像和朋友聊天一样，用一两句话自然点评整体表现，提到得分和等级。不要用"总体评价"这种生硬标题。

### 2. 亮点
像夸奖学生一样，列出2-3个闪光点。引用具体身体部位和数据，让学生感受到你真的看懂了TA的动作。

### 3. 可以更棒的地方
用"可以更棒"的口吻（而非"需要改进"），给出2-3条具体可操作的小建议，每条附上练习方法。

### 4. 教练寄语
用温暖有力的语言结尾，像真正的舞蹈老师那样激励学生。强调进步的空间和坚持练习的价值，不要使用"鼓励语""建议"这样的生硬标签。
"""
    return prompt


# ---------------------------------------------------------------------------
# 2. Gemini Integration
# ---------------------------------------------------------------------------

def evaluate_with_gemini(
    api_key: str,
    prompt: str,
    model: str = "gemini-2.0-flash",
) -> str:
    """
    Send the prompt to Gemini API and return the coaching report.

    Uses the google-generativeai SDK.
    On any error, returns a friendly Chinese error message.
    """
    if not api_key:
        return "错误：未提供 Gemini API 密钥，请设置后重试。"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)

        if response and hasattr(response, "text") and response.text:
            return response.text
        else:
            return "错误：Gemini API 返回了空结果，请稍后重试。"

    except Exception as e:
        error_msg = str(e)
        # Provide a more user-friendly message for common errors
        if "API_KEY" in error_msg.upper() or "API key" in error_msg:
            return "错误：Gemini API 密钥无效或已过期，请检查后重试。"
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return "错误：Gemini API 调用次数已达限额，请稍后重试。"
        elif "timeout" in error_msg.lower():
            return "错误：Gemini API 请求超时，请检查网络后重试。"
        else:
            return f"错误：调用 Gemini API 时发生异常——{error_msg}"


# ---------------------------------------------------------------------------
# 3. DeepSeek Integration
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "你是一位拥有20年经验的专业舞蹈教练，擅长分析舞蹈动作并给出专业、"
    "具体且充满鼓励的点评。请始终用中文回复，语气温暖而专业。"
)


def evaluate_with_deepseek(
    api_key: str,
    prompt: str,
    model: str = "deepseek-chat",
) -> str:
    """
    Send the prompt to DeepSeek API and return the coaching report.

    Uses the openai SDK with base_url pointing to DeepSeek.
    On any error, returns a friendly Chinese error message.
    """
    if not api_key:
        return "错误：未提供 DeepSeek API 密钥，请设置后重试。"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        content = response.choices[0].message.content
        if content:
            return content
        else:
            return "错误：DeepSeek API 返回了空结果，请稍后重试。"

    except Exception as e:
        error_msg = str(e)
        if "api" in error_msg.lower() and "key" in error_msg.lower():
            return "错误：DeepSeek API 密钥无效或已过期，请检查后重试。"
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return "错误：DeepSeek API 调用次数已达限额，请稍后重试。"
        elif "timeout" in error_msg.lower():
            return "错误：DeepSeek API 请求超时，请检查网络后重试。"
        else:
            return f"错误：调用 DeepSeek API 时发生异常——{error_msg}"


# ---------------------------------------------------------------------------
# 4. Main Entry Point
# ---------------------------------------------------------------------------

def generate_coaching_report(
    api_key: str,
    provider: str,
    metrics: dict,
    model: Optional[str] = None,
) -> str:
    """
    Generate a Chinese narrative coaching report from dance metrics.

    Args:
        api_key: API key for the chosen provider.
        provider: "gemini" or "deepseek".
        metrics: Structured dict from metrics.compute_all_metrics().
        model: Override default model name. Defaults are:
               "gemini-2.0-flash" for Gemini, "deepseek-chat" for DeepSeek.

    Returns:
        Coaching report text in Chinese. On error or empty api_key,
        returns a Chinese error message — never crashes.
    """
    try:
        prompt = build_coaching_prompt(metrics)
    except Exception as e:
        return f"错误：构建提示词时发生异常——{str(e)}"

    provider = provider.lower().strip()

    if provider == "gemini":
        default_model = "gemini-2.0-flash"
        chosen_model = model or default_model
        return evaluate_with_gemini(api_key, prompt, chosen_model)
    elif provider == "deepseek":
        default_model = "deepseek-chat"
        chosen_model = model or default_model
        return evaluate_with_deepseek(api_key, prompt, chosen_model)
    else:
        return f"错误：不支持的提供商 '{provider}'，请使用 'gemini' 或 'deepseek'。"


# ---------------------------------------------------------------------------
# 5. Test Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    try:
        from video_processor import process_video_pair
        from metrics import compute_all_metrics
    except ImportError as e:
        print(f"无法导入依赖模块：{e}")
        print("请确保 video_processor.py 和 metrics.py 在当前目录下。")
        sys.exit(1)

    print("=" * 60)
    print("舞蹈动作分析 - LLM 教练点评生成器")
    print("=" * 60)

    # Step 1 — Process videos
    print("\n[1/3] 正在处理视频...")
    try:
        data = process_video_pair("1teacher.mp4", "1me.mp4", num_frames=30)
        print(f"  -> 处理完成，有效帧数: {data.get('num_valid_frames', '?')}")
    except FileNotFoundError as e:
        print(f"  -> 视频文件未找到：{e}")
        print("  请将 '1teacher.mp4' 和 '1me.mp4' 放在当前目录后重试。")
        sys.exit(1)

    # Step 2 — Compute metrics
    print("\n[2/3] 正在计算舞蹈指标...")
    metrics = compute_all_metrics(data)
    overall = metrics.get("overall_score", 0)
    print(f"  -> 综合得分: {overall:.1f}")

    # Step 3 — Generate coaching report
    print("\n[3/3] 正在生成教练点评报告...")
    api_key = "YOUR_KEY_HERE"

    print("\n--- Gemini Report ---")
    gemini_report = generate_coaching_report(api_key, "gemini", metrics)
    print(gemini_report)

    print("\n--- DeepSeek Report ---")
    deepseek_report = generate_coaching_report(api_key, "deepseek", metrics)
    print(deepseek_report)

    print("\n" + "=" * 60)
    print("测试完成。请将 'YOUR_KEY_HERE' 替换为真实的 API 密钥以获取实际报告。")
    print("=" * 60)
