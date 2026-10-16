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

    python run.py                      # 语音模式(麦克风 + 扬声器)
    python run.py --text               # 文本模式(不起声卡, 不加载 KWS)
    python run.py --text --with-tts    # 文本输入, 但保留真实 TTS 合成链路
    python run.py --ask "现在几点了"    # 问一句就退出(脚本/排查用)

   文本模式复用**同一条** pipeline.invoke, 只是把"音频从哪来"换成 stdin ——
   用来在没有麦克风/没配 STT 的环境下验证链路逻辑(见 transport/text.py)。

Configuration
-------------
Edit the parameters below to match your setup.
"""

import argparse
import asyncio
import os
import sys
import time
# Make the package importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant import VoiceAssistant
from voice_assistant.backend.adapter import AudioDeviceAdapter
from voice_assistant.backend.kws import KeywordWakeDetector
from voice_assistant.perf import PerfRecorder
from voice_assistant.transport import TransportRegistry
from voice_assistant.transport.local import LocalTransport
from voice_assistant.transport.text import TextTransport


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

# Delegate (复杂任务): 常驻中继插件跑在 qqbot profile 进程内(见 qqbot_relay)
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
    os.path.dirname(os.path.abspath(__file__)), "data", "metrics", "perf_baseline.jsonl"
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

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="语音助手（默认语音模式；--text 走文本模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--text", action="store_true",
                   help="文本交互模式：不起声卡、不加载 KWS，直接打字对话")
    p.add_argument("--ask", metavar="问题",
                   help="问一句就退出（隐含 --text），适合脚本与远程排查")
    p.add_argument("--with-tts", action="store_true",
                   help="文本模式下仍执行真实 TTS 合成（默认关闭：文本模式只需 DEEPSEEK_API_KEY）")
    p.add_argument("--wait", type=float, default=0.0, metavar="秒",
                   help="--ask 之后等后台委托任务跑完（并播出进度/结果），默认 0=不等")
    # --- 阶段 1：手机/网页远端接入 ---
    p.add_argument("--serve-remote", action="store_true",
                   help="启动远端服务(WebSocket + 网页测试端)，手机浏览器可直接连")
    p.add_argument("--port", type=int, default=8801, help="远端服务端口(默认 8801)")
    p.add_argument("--host", default="0.0.0.0", help="远端服务监听地址(默认 0.0.0.0)")
    p.add_argument("--token", default=None, metavar="口令",
                   help="远端连接口令；不传则自动生成一个随机口令（调试期嫌麻烦就用 --no-auth）")
    p.add_argument("--no-auth", action="store_true",
                   help="⚠️ 不校验口令：手机端不用填口令，直接连（同网段任何人都能连，仅限调试）")
    p.add_argument("--no-local-mic", action="store_true",
                   help="不开本机麦克风（只服务远端；本机没有声卡时用）")
    p.add_argument("--tls", action="store_true",
                   help="远端服务启用 https/wss（自动生成自签证书）。手机浏览器只在安全上下文里给麦克风，http://局域网IP 会被直接拒绝")
    p.add_argument("--no-tls", action="store_true", help="强制关闭 TLS（仅调试用）")
    p.add_argument("--tls-dir", default=None, metavar="目录",
                   help="自签证书目录（默认 data/certs，首次自动生成并复用）")
    p.add_argument("--wait-connection", type=float, default=0.0, metavar="秒",
                   help="启动后等手机连上（打印手机可用的地址）再进入监听，默认 0=不等")
    return p.parse_args(argv)


def _local_ip() -> str:
    """取本机在局域网里的 IPv4（给手机用的地址）。

    用 UDP connect 探测默认出口 —— 不会真的发包，也不需要 DNS。
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 53))
        return s.getsockname()[0]
    except Exception:                                             # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


async def main():
    args = _parse_args()
    text_mode = args.text or bool(args.ask)
    # 文本模式默认不合成：这样"只验证链路逻辑"时连 DASHSCOPE_API_KEY 都不需要。
    # 但代码路径不变 —— 工具轮次/委托/承诺纠正/落历史都照跑，只是不调 TTS。
    synthesize = (not text_mode) or args.with_tts

    # --- 先检查 key, 免得跑起来才在对话中途拿到 401 ---
    needed = [("DEEPSEEK_API_KEY", LLM_API_KEY)]
    if synthesize:
        needed.append(("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY))
    missing = [name for name, val in needed if not val]
    if missing:
        print("❌ 缺少环境变量: " + ", ".join(missing))
        print("   设置方式(PowerShell):")
        for name in missing:
            print(f'     $env:{name}="sk-..."')
        print("   或者直接填到本文件顶部的 DASHSCOPE_API_KEY / LLM_API_KEY 默认值里")
        if text_mode and "DASHSCOPE_API_KEY" in missing:
            print("   提示: 文本模式默认不合成语音, 所以只需要 DEEPSEEK_API_KEY;")
            print("        若想连 TTS 一起验证, 再加 --with-tts 并配好 DASHSCOPE_API_KEY")
        return

    t_start = time.perf_counter()

    # --- KWS (local keyword spotting) ---
    # 模型加载要 ~8s(纯 CPU), 丢到线程里和下面的 pipeline 构造并行, 启动少等 8s。
    # 文本模式不经过语音, 加载它纯属浪费(还会占 8s 启动时间), 直接跳过。
    kws = None
    kws_task = None
    if KWS_ENABLED and not text_mode:
        print(f"🔑 Loading local KWS model for keywords: {KWS_KEYWORDS}")

        def _load_kws():
            return KeywordWakeDetector(
                keywords=KWS_KEYWORDS,
                model_dir=KWS_MODEL_DIR,
                keywords_threshold=KWS_THRESHOLD,
                debug=True,
            )

        kws_task = asyncio.create_task(asyncio.to_thread(_load_kws))
    elif text_mode:
        print("⌨️  文本模式: 跳过 KWS 模型加载(不经过语音)")
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
        # 无头(文本)模式: 不构造需要凭据的 STT/TTS —— 否则"只想跑文本逻辑"会因为
        # 缺 DASHSCOPE_API_KEY 直接崩在启动阶段（实测踩到过）。
        text_only=text_mode and not synthesize,
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
    print(f"📊 时延基线将按路径累积到 {PERF_BASELINE_PATH}")

    # ⚠️ 这一行必须放在模式分支**之前**（两种模式都要落盘）。
    # 实测踩坑：加文本分支时把它挪进了 `if text_mode:` 里，于是**语音路径的回调
    # 退回默认处理器** —— 只打印 [PERF] 一行、一条样本都不写盘，退出时还报
    # "本次没有样本"。日志看着正常，很容易漏掉，所以这里显式标注。
    assistant.on_performance(perf.record)

    # ==================================================================
    # 文本模式：同一套 pipeline / transport 装配, 只是音频端换成 stdin
    # ==================================================================
    if text_mode:
        text_t = TextTransport(assistant, synthesize=synthesize)
        registry = TransportRegistry(default=text_t)
        assistant.attach_transport(registry)
        # 文本端没有 VAD/STT, 那几个以"判句时刻"为起点的指标必然为空;
        # 这里改用"提问时刻"为起点, 并把 pipeline 的指标 dict 原样透传下去。
        text_t._prev_perf = perf.record
        assistant.on_performance(text_t._on_perf)

        try:
            if args.ask:
                await text_t.ask(args.ask)
                # 委托是异步的：不等的话"确认话术"之后就退出了，结果与播报全丢
                if args.wait > 0:
                    await text_t.wait_for_background(timeout=args.wait)
            else:
                # 交互模式默认等 60s：委托是异步的，不等就会"确认话术之后就没下文"
                await text_t.run(wait_background=args.wait if args.wait > 0 else 60.0)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            await registry.close()
            await assistant.shutdown()
            perf.print_summary()
        return

    # --- Wire up audio I/O (传输层: 本机声卡端 + 可选远端) ---
    # 阶段0 起音频 I/O 走 transport 抽象: 本机端只是把 AudioDeviceAdapter 包了一层,
    # 行为与改造前逐字一致(见 transport/local.py)。手机端 = 再加一个 RemoteTransport,
    # 注册进同一个 registry 即可, pipeline 无需改动。
    local_t = None
    if args.no_local_mic:
        print("🔇 按参数要求不开本机麦克风（只服务远端）")
    else:
        local_t = LocalTransport(AudioDeviceAdapter(assistant, cancel_echo=ECHO_CANCELLATION))
    registry = TransportRegistry(default=local_t)
    assistant.attach_transport(registry)

    # --- 阶段 1: 远端服务(WebSocket + 网页测试端) ---
    remote_t = None
    if args.serve_remote:
        from voice_assistant.transport.websocket import RemotePerf, RemoteTransport
        import secrets

        # 口令：默认自动生成；`--no-auth` 或 `--token ""` 可关掉（调试期更省事）
        token = "" if args.no_auth else (args.token or secrets.token_urlsafe(6))
        perf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "metrics", "perf_remote.jsonl")
        remote_t = RemoteTransport(
            assistant, host=args.host, port=args.port, token=token,
            serve_web=True,
            web_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"),
            perf=RemotePerf(perf_path, debug=True),
            # TLS 默认**开**（--serve-remote 时）：手机浏览器的麦克风只在安全上下文
            # 可用，http://局域网IP 会被直接拒绝 —— 这不是"更安全"的可选项，是可用性前提。
            # 需要临时关掉用 --no-tls。
            tls=(not args.no_tls),
            tls_dir=args.tls_dir,
            tls_hosts=[_local_ip()],
            debug=True,
        )
        registry.register(remote_t)
        try:
            await remote_t.start_server()
        except OSError as ex:
            print(f"❌ 远端服务启动失败（端口 {args.port} 被占用？）: {ex}")
            return
        except RuntimeError as ex:
            print(f"❌ TLS 启用失败: {ex}")
            print("   可以先用 --no-tls 跑起来（但手机端麦克风会被浏览器拒绝）")
            return
        ip = _local_ip()
        scheme = "https" if remote_t.tls else "http"
        print("\n" + "=" * 66)
        print("📱 手机端（同一局域网内打开）:")
        print(f"     {scheme}://{ip}:{args.port}/")
        if remote_t.tls:
            print(f"   证书指纹: {remote_t.tls_fingerprint}")
            print("   ⚠️ 自签证书：手机会提示不安全，选「高级 → 继续访问」即可（仅首次）")
        if token:
            print(f"   口令: {token}")
        else:
            print("   ⚠️ 未设口令（--no-auth）：同网段任何人打开这个地址都能说话")
            print("      —— 适合调试期图省事；将来挂公网入口前务必打开口令")
        print("=" * 66 + "\n")

    # 可选：等手机连上再进入监听（方便先看到地址/确认联通）
    if remote_t is not None and args.wait_connection > 0:
        print(f"⏳ 等待远端连接（最多 {args.wait_connection:.0f}s）…")
        deadline = time.perf_counter() + args.wait_connection
        while time.perf_counter() < deadline:
            if remote_t.stats()["connections"] > 0:
                print("✅ 手机已连上")
                break
            await asyncio.sleep(0.5)
        else:
            print("（没等到连接，继续启动；手机随时可以连）")

    # --- Start listening (blocks until Ctrl+C) ---
    print(f"\n🎤 Voice assistant ready (启动共 {time.perf_counter() - t_start:.1f}s). "
          f"Press Ctrl+C to stop.\n")
    try:
        if local_t is not None:
            await local_t.start("default_user", "default_user")
        else:
            # 只服务远端：没有本机麦克风可听，就等 Ctrl+C
            print("（未开本机麦克风，等待远端连接…）")
            await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        await registry.close()
        await assistant.shutdown()
        perf.print_summary()


if __name__ == "__main__":
    asyncio.run(main())

