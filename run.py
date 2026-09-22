#!/usr/bin/env python3
"""Run the voice assistant with microphone input and speaker output.

Prerequisites
-------------
1. Set environment variables::

    export DASHSCOPE_API_KEY="your_aliyun_dashscope_key"   # STT + CosyVoice TTS
    export DEEPSEEK_API_KEY="your_deepseek_api_key"        # LLM (直连, 不需要 nanobot)

   For Baidu TTS (legacy), also set::

    export BAIDU_TTS_API_KEY="your_baidu_api_key"
    export BAIDU_TTS_SECRET_KEY="your_baidu_secret_key"

2. Run::

    python run.py

Configuration
-------------
Edit the parameters below to match your setup.
"""

import asyncio
import os
import sys
import time

# Make the package importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant import VoiceAssistant
from voice_assistant.adapter import AudioDeviceAdapter
from voice_assistant.kws import KeywordWakeDetector
from voice_assistant.perf import PerfRecorder


# ================================================================
# Configuration — adjust these to your environment
# ================================================================

# STT: Aliyun DashScope
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
STT_LANGUAGE = "zh"

# TTS: CosyVoice v3-flash(百炼, 流式首包 ~0.6s, 和 STT 用同一个 key)
# 备选: "edge"(免费, 但每句要等整句 mp3 + ffmpeg 转码)、"baidu"(legacy)
TTS_PROVIDER = "cosyvoice"
# CosyVoice (only used if TTS_PROVIDER="cosyvoice")
TTS_MODEL = "cosyvoice-v3-flash"
TTS_VOICE = "longanhuan"  # 龙安欢 (cheerful female)
# Edge TTS (only used if TTS_PROVIDER="edge")
EDGE_TTS_VOICE = "zh-CN-YunyangNeural"
# 其他中文音色: zh-CN-YunxiNeural(活泼男) zh-CN-YunjianNeural(沉稳男)
#              zh-CN-XiaoyiNeural(女) zh-CN-liaoning-XiaobeiNeural(东北话)
EDGE_TTS_RATE = "+10%"   # 语速, 如 "+10%" / "-20%"
EDGE_TTS_PITCH = "+0Hz"  # 音调
EDGE_TTS_VOLUME = "+0%"  # 音量
# Baidu TTS fallback (only used if TTS_PROVIDER="baidu")
BAIDU_TTS_API_KEY = os.getenv("BAIDU_TTS_API_KEY", "")
BAIDU_TTS_SECRET_KEY = os.getenv("BAIDU_TTS_SECRET_KEY", "")
TTS_SPEAKER = "1"  # Baidu speaker id

# LLM: 直连 DeepSeek 的 OpenAI 兼容接口(原来的 nanobot 网关已移除, 历史在本地维护)
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-flash"
LLM_THINKING = False  # 语音场景首字优先: 关掉思考模式, 跳过思维链
LLM_SYSTEM_PROMPT = (
    "你叫小牛，是一个友好的语音助手，用简洁干练的中文回答用户的问题。"
    "回答只输出适合朗读的口语化纯文字，严禁使用 Markdown 符号（如 **、`、#、- 列表）、"
    "表情符号或任何符号装饰。回答要短，一般一到两句话。\n"
    "调用任何工具之前，先输出一句极短的确认话术（例如「好的，我去查一下」「稍等，我看看」），"
    "然后再发起工具调用，不要只说工具不说人话。\n"
    "涉及实时行情、时效性数据（今天/最新/现在 + 数值）、时政新闻、多步分析或报告生成时，"
    "必须调用 delegate_to_dsh 交给后台处理，不要凭记忆回答；调用后本轮就结束，"
    "结果由系统稍后播报，你不需要等待也不要追问。"
)

# VAD
VAD_VOLUME_THRESHOLD = -20  # dB, lower = more sensitive
VAD_SILENCE_THRESHOLD = 0.5  # seconds

# Wake / interaction
WAKEWORDS = ["小牛"]  # Set to None to always stay awake
#
# END_WORDS 的收录原则：**只收"本义就是停止/离开"的说法**。
#
# 判据只有一条朴素的子串匹配（见 pipeline._has_end_word），所以凡是会出现在
# 普通请求/陈述里的日常词都必须剔除 —— 它们会把正常对话误判成"结束"，助手于是
# 静默回到待唤醒态（用户以为程序退出）。2026-09-17 实机就踩过：
#   「天气查好了吗？」命中"好了" → awake ended，之后不喊唤醒词就不应答。
#
# 已剔除的歧义词（及原因），不要再加回来：
#   好了      「查好了吗」「做好了」           ← 本次事故元凶
#   行了      「就行了吗」（行了吧 保留，语气明确）
#   好嘞      日常应答（"好嘞，谢谢"），不是结束
#   了解      「了解一下…」是常见请求
#   收到      「收到消息」；口语应答也常说
#   懂了/知道了/已阅   陈述性应答，未必表示结束
#   可以了    「这样可以了吗」
#   够了      「钱够了」「够了没」
#   下去      **语义反转**：「说下去」= 让你继续
#   你继续    **语义反转**：「你继续说」= 让你继续
#   就这样    「就这样做吧」= 按这个方式继续（"就这样吧" 保留）
#   歇着      「歇着吧」已保留
#   停/拜     单字，误伤「停车」「拜年」
#   结束      「会议结束了」等陈述
END_WORDS = [
    # 告别
    "再见", "拜拜", "回头见", "下次见", "一会见", "晚点聊",
    "明天见", "晚安", "回见", "走啦", "先走了",
    # 明确要求安静 / 停止
    "别说了", "住口", "住嘴", "闭嘴", "别吵了", "别吵", "别闹了",
    "别念叨了", "别逼逼了", "停下", "停止", "打住",
    # 退下 / 赶人
    "退下", "下去吧", "没你事了", "你可以走了", "你走吧",
    "你可以退下了", "跪安", "散了吧", "退开", "一边去", "闪一边去",
    # 去休息 / 去玩（"退下" 已覆盖 "退下吧"/"你退下" 等变体）
    "去玩吧", "去休息吧", "歇着吧", "睡觉去吧", "睡吧",
    "休息吧", "去歇着", "去睡觉", "你睡吧",
    # 明确表示不需要了
    "不用了", "没事了", "就这样吧", "先这样", "先到这", "就到这",
    "到这吧", "差不多了", "行了行了", "好了好了", "行了吧", "得了吧",
    # 闲聊式结束
    "忙你的去吧", "你忙吧", "不打扰你了", "朕知道了", "爱卿退下",
]
AWAKE_TIMEOUT = 600.0  # seconds
CONTEXT_TIMEOUT = 3600.0  # 1 hour

# Noise filter (Plan D): short segments (< MIN_MEANINGFUL_DURATION) whose
# recognized text is only a filler word are silently ignored.
MIN_MEANINGFUL_DURATION = 1.0  # seconds
NOISE_WORDS = [
    "嗯", "嗯嗯", "嗯呢", "嗯嗯嗯",
    "哦", "哦哦", "噢", "喔",
    "啊", "额", "呃", "诶", "哎", "唉",
    "嘛", "呢", "吧", "呀", "呐",
    "哈", "呵", "嘿", "嗨", "嘻",
    "对", "是的", "好的", "好吧", "好",
]

# Audio device
ECHO_CANCELLATION = True  # Mute mic while playing AI speech

# Delegate (复杂任务): 常驻中继插件跑在 qqbot profile 进程内(见 bt_assistant/qqbot_relay)
DELEGATE_URL = "http://127.0.0.1:8765"   # 中继控制面(仅本机监听)
DELEGATE_ENABLED = True
DELEGATE_PROGRESS_AFTER = 25.0           # 用户等待超过这么多秒才播一次进度
DELEGATE_PROGRESS_INTERVAL = 45.0        # 之后每隔这么久最多再播一次
# 取消后台查询的本地词表(命中即掐掉任务, 不过 LLM)
CANCEL_WORDS = [
    "别查了", "不用查了", "不查了", "停止查询", "取消查询", "别查",
    "算了别查", "先不查了",
]

# 时延基线: 每轮的时延按路径(chat/tool/delegate)累积落盘, Ctrl+C 退出时打印汇总
PERF_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bt_assistant", "perf_baseline.jsonl"
)

# KWS (local keyword spotting with sherpa-onnx)
# Set to None to disable and fall back to cloud-STT-based wake word detection
KWS_ENABLED = True
KWS_KEYWORDS = ["小牛"]       # Keywords to detect locally
KWS_MODEL_DIR = None          # None = use the default bundled model
KWS_THRESHOLD = 0.25          # Detection confidence threshold (lower = more sensitive)


# ================================================================
# Main
# ================================================================

async def main():
    # --- 先检查 key, 免得跑起来才在对话中途拿到 401 ---
    missing = [name for name, val in (("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY),
                                      ("DEEPSEEK_API_KEY", LLM_API_KEY)) if not val]
    if missing:
        print("❌ 缺少环境变量: " + ", ".join(missing))
        print("   设置方式(PowerShell):")
        for name in missing:
            print(f'     $env:{name}="sk-..."')
        print("   或者直接填到本文件顶部的 DASHSCOPE_API_KEY / LLM_API_KEY 默认值里")
        return

    t_start = time.perf_counter()

    # --- KWS (local keyword spotting) ---
    # 模型加载要 ~8s(纯 CPU), 丢到线程里和下面的 pipeline 构造并行, 启动少等 8s
    kws = None
    kws_task = None
    if KWS_ENABLED:
        print(f"🔑 Loading local KWS model for keywords: {KWS_KEYWORDS}")

        def _load_kws():
            return KeywordWakeDetector(
                keywords=KWS_KEYWORDS,
                model_dir=KWS_MODEL_DIR,
                keywords_threshold=KWS_THRESHOLD,
                debug=True,
            )

        kws_task = asyncio.create_task(asyncio.to_thread(_load_kws))
    else:
        print("⚠️  KWS disabled — using cloud STT for wake-word detection (not cost-effective)")

    # --- Build pipeline ---
    assistant = VoiceAssistant(
        kws=None,   # 稍后填进去
        stt_api_key=DASHSCOPE_API_KEY,
        stt_language=STT_LANGUAGE,
        tts_api_key=BAIDU_TTS_API_KEY if TTS_PROVIDER == "baidu" else DASHSCOPE_API_KEY,
        tts_provider=TTS_PROVIDER,
        tts_model=TTS_MODEL,
        tts_voice=EDGE_TTS_VOICE if TTS_PROVIDER == "edge" else TTS_VOICE,
        tts_rate=EDGE_TTS_RATE if TTS_PROVIDER == "edge" else None,
        tts_pitch=EDGE_TTS_PITCH if TTS_PROVIDER == "edge" else None,
        tts_volume=EDGE_TTS_VOLUME if TTS_PROVIDER == "edge" else None,
        tts_secret_key=BAIDU_TTS_SECRET_KEY,  # only used if provider="baidu"
        tts_speaker=TTS_SPEAKER,
        llm_api_key=LLM_API_KEY,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        llm_thinking=LLM_THINKING,
        llm_system_prompt=LLM_SYSTEM_PROMPT,
        vad_volume_threshold=VAD_VOLUME_THRESHOLD,
        vad_silence_threshold=VAD_SILENCE_THRESHOLD,
        wakewords=WAKEWORDS,
        end_words=END_WORDS,
        noise_words=NOISE_WORDS,
        min_meaningful_duration=MIN_MEANINGFUL_DURATION,
        awake_timeout=AWAKE_TIMEOUT,
        context_timeout=CONTEXT_TIMEOUT,
        # 复杂任务委托(常驻中继)
        delegate_url=DELEGATE_URL,
        delegate_enabled=DELEGATE_ENABLED,
        delegate_progress_after=DELEGATE_PROGRESS_AFTER,
        delegate_progress_interval=DELEGATE_PROGRESS_INTERVAL,
        cancel_words=CANCEL_WORDS,
        debug=True,
    )

    # --- 预热: 建 LLM 的 httpx/SSL 上下文 + TTS 首条连接 ---
    # (不预热的话这几秒会落在第一轮对话的首字延时里)
    # 和 KWS 模型加载并行跑, 启动时间 ≈ max(KWS 加载, 预热)
    t_warm = time.perf_counter()
    warm_task = asyncio.create_task(assistant.warmup())
    if kws_task is not None:
        assistant.kws = await kws_task
        print("✅ KWS ready (offline wake-word detection)")
    await warm_task
    print(f"🔥 warmup {time.perf_counter() - t_warm:.1f}s")

    # --- 时延基线采集(按路径分组) ---
    perf = PerfRecorder(PERF_BASELINE_PATH, debug=True)
    assistant.on_performance(perf.record)
    print(f"📊 时延基线将按路径累积到 {PERF_BASELINE_PATH}")

    # --- Wire up audio I/O ---
    adapter = AudioDeviceAdapter(assistant, cancel_echo=ECHO_CANCELLATION)

    # --- Start listening (blocks until Ctrl+C) ---
    print(f"\n🎤 Voice assistant ready (启动共 {time.perf_counter() - t_start:.1f}s). "
          f"Press Ctrl+C to stop.\n")
    try:
        await adapter.start_listening("default_user")
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        adapter.close()
        await assistant.shutdown()
        perf.print_summary()


if __name__ == "__main__":
    asyncio.run(main())

