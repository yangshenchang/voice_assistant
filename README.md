# voice_assistant · 小牛语音助手

一个**本地优先**的中文语音助手：唤醒词 → 语音识别 → 大模型 → 语音合成，
复杂任务异步委托给后台 agent（进度/结果主动播报，全文推 QQ），
并支持**手机浏览器**作为音频端接入。

> ⚠️ 这个目录**本身就是 Python 包 `voice_assistant`**（仓库名即包名）。
> 所有 `from voice_assistant.xxx import ...` 都依赖这一点，父子目录的 import 也走包内相对导入。
> **不要把包再往下嵌一层**，否则全工程导入会断。

---

## 一、模块划分（按功能）

```
voice_assistant/                     ← 仓库根 = 包 voice_assistant
│
├─ run.py                    ★ 入口脚本（语音 / 文本 / 远端服务三种模式）
│
├─ 【编排层】根目录
│   ├─ pipeline.py           ★ 编排核心：VAD→STT→LLM(工具轮次)→TTS，唤醒/打断/播报状态机
│   ├─ delegate.py           ★ 委托客户端 + 播报策略（提交即走、进度累计、结果播报、取消）
│   ├─ tools.py                内置工具（时间）+ 工具返回约定（ASYNC_TOOL）
│   ├─ perf.py                 时延采集
│   └─ models.py  context_manager.py   数据契约（与能力无关，各层共享）
│
├─ backend/                  ◆ 模型与设备能力层（"用什么去听/去说/去想"）
│   ├─ adapter.py               本机声卡 I/O（PyAudio；麦克风采集线程 + 扬声器播放）
│   ├─ stt.py                  语音识别（DashScope qwen3-asr-flash）+ 无头占位实现
│   ├─ tts.py                  语音合成（CosyVoice 流式 / Edge / 百度）+ 无头占位实现
│   ├─ llm.py                  OpenAI 兼容流式 LLM（tool_calls 累加、切句、TTS 文本清洗）
│   ├─ vad.py                  振幅阈值 VAD（判句）
│   └─ kws.py                  本地唤醒词（sherpa-onnx 流式）
│
├─ transport/                ◆ 音频端抽象（让编排层只认接口，不认设备）
│   ├─ __init__.py             AudioTransport 接口 + TransportRegistry（路由/仲裁/发声）
│   ├─ local.py                本机麦克风+扬声器（包装 backend/adapter）
│   ├─ text.py                 文本端（stdin/stdout，无需声卡与 STT）
│   ├─ remote.py               远端基类（录音窗/播放态/中断回报）
│   ├─ websocket.py            手机接入服务端：WSS + 静态页 + 鉴权 + 状态机
│   └─ tls.py                  自签证书（浏览器只在安全上下文给麦克风）
│
├─ protocol/__init__.py      ◆ 协议层：纯编解码 + 常量（手机端与将来 App 共用）
├─ web/                      ◆ 手机网页端（按住说话）
│   ├─ index.html  app.js  capture-worklet.js
│
├─ qqbot_relay/              ▲ QQ 侧：常驻在 qqbot profile 内的委托中继插件
│   ├─ index.js                 HTTP 控制面 + 受限 dsh agent + 进度回流 + 完成推 QQ
│   ├─ qq_push.js               QQ 官方主动推送（零依赖，token 缓存 + 分块）
│   ├─ cordis.patch.yml / package.json / README.md
│
├─ docs/                     ■ 设计与实测记录
│   ├─ 手机外网访问实现.md       **实现文档**：手机接入机制/协议/状态机/TLS/参数速查/扩展点
│   ├─ 传输层规划.md             音频端与手机接入的**规划与踩坑记录**（16 个真 bug）
│   ├─ 混合路由规划.md           架构主线 + 委托（含各 Phase 实测）
│   ├─ 低时延规划.md             时延专项（含基线数字）
│   └─ 历史方案-realtime混合架构.md  早期单 WS 端到端方案（已废弃，仅存档）
│
├─ tools/probe_ingress.py     □ 诊断：公网入口（DNS / 端口，只读）
├─ tools/probe_external.py    □ 诊断：本机外网可达性（公网 IP / 端口 / IPv6 / UPnP，只读）
├─ tools/probe_relay_target.py □ 诊断：公网入口会不会转发到本机（探针 + 探测）
├─ tests/                    □ 验证脚本（见第三节）
├─ legacy/                   □ 历史存档（不参与现链路）
│   ├─ realtime_bridge.py       旧"realtime 单 WS"桥接层
│   └─ fastest_demo.py          当初提炼低时延策略的极简 demo
└─ data/                     □ 运行时产物（不进版本控制）
    ├─ metrics/                 时延基线 jsonl
    └─ certs/                   自签证书（私钥！已 gitignore）
```

**分层判据**：`backend/` 只提供**能力**（一次识别、一次合成、一次补全），
不含任何"谁先谁后、何时开口、失败怎么办"——那些属于编排层（`pipeline` / `delegate` / `transport`）。

三层数据流：

```
本机麦克风 ─┐
手机 wss ──┼─→ transport ─→ pipeline(VAD→STT→LLM→TTS) ─→ transport ─→ 出声
文本 stdin ─┘       │              │
                    │              └─ backend/：设备与模型能力
                    └─ 复杂任务 → qqbot_relay（受限 agent）
                                     └─ 结果：语音播报 + QQ 全文
```

> **兼容性说明**：`voice_assistant.vad` / `.llm` / `.stt` 这类**旧导入路径仍然可用**
> （由 `backend/__init__.py` 的 `install_legacy_aliases()` 自动指向 `backend.*`），
> 以免历史脚本因为这次搬运而断掉。**新代码请直接用 `voice_assistant.backend.xxx`。**

---

## 二、怎么跑

### 语音模式（本机麦克风 + 扬声器）

```powershell
python run.py
```

### 手机接入（局域网，按住说话）

```powershell
python run.py --serve-remote --wait-connection 120     # 默认开 TLS、自动生成口令
#   --no-auth        不校验口令（调试期省事；挂公网前务必打开）
#   --no-local-mic   不开本机麦克风（只服务手机）
#   --port 8801      端口
```

启动后按提示在手机浏览器打开 `https://<本机IP>:8801/`，首次点「高级 → 继续访问」信任自签证书。

> **外网访问**：已通过 **cpolar 隧道**实测跑通（手机 4G，网络 RTT **96 ms**、丢帧 **0**、
> 连续通话 1.5 分钟不断）—— 机制、协议、状态机、参数与实测数据见
> **`docs/手机外网访问实现.md`** 与 **`docs/cpolar外网联调清单.md`**。
>
> ⚠️ 公网部署**必须开语音口令**（去掉 `--no-auth`），并注意**不要**把 RDP/3080 之类的
> 端口挂到隧道上 —— 安全清单见 `docs/cpolar外网联调清单.md` 第五节。

### 文本模式（无麦克风/调试链路）

```powershell
python run.py --text                     # 交互
python run.py --ask "现在几点了"           # 问一句就退
python run.py --ask "查今天上证指数" --wait 300   # 等后台委托结果
```

### 依赖与凭据

```powershell
pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="sk-..."     # STT + CosyVoice TTS
$env:DEEPSEEK_API_KEY="sk-..."      # LLM（也可直接写在 run.py 顶部的默认值里）
```

---

## 三、各部分怎么验证

| 目录/文件 | 用途 | 需要什么 |
|---|---|---|
| `tests/offline/` | **离线验收**：传输层/文本入口/服务端协议与状态机/TLS | 不需要声卡、不需要网络、不需要 key |
| `tests/phone/` | 手机链路：模拟客户端、按下时序复现、合成测试语音 | 需要服务端在跑 |
| `tests/voice/` | 语音链路（文本驱动，不碰音频设备） | 需要 key |
| `tests/relay/` | 委托中继：控制面驱动、ACP 探测、headless 探测 | 需要中继在线 |
| `tests/latency/` | 时延专项实验（历史结论的复现脚本） | 需要 key |
| `tools/probe_ingress.py` | 公网入口诊断（DNS/端口，只读） | 无 |

常用：

```powershell
python tests/offline/f2_remote_server.py     # 服务端协议+状态机（101 项）
python tests/offline/f5_tls.py               # 自签证书与 HTTPS/WSS（24 项）

# 手机链路时序复现（服务端在跑时）：先合成一句真语音，再模拟手机推上去
python tests/phone/make_prompt_wav.py "现在几点了" -o "$env:TEMP\ask.wav"
python tests/phone/f6_press_timing.py --tls --audio "$env:TEMP\ask.wav"
```

`tests/` 已加入 `.gitignore`（属验证工具，非产品代码）。

---

## 四、QQ 侧（委托中继）

`qqbot_relay/` 是一个 **dsh Cordis 插件**，装在 `qqbot` profile 里：

* 提供本机 HTTP 控制面（默认 `127.0.0.1:8765`）：`/health` `/task` `/task/:id` `/cancel`
* 驱动一个**工具受限**的 dsh agent（白名单：`web_search`/`web_fetch`/`read`/`glob`/`grep`）
* 把完整报告主动推到 QQ（`qq_push.js`），口语结论回流给语音侧播报

安装与自检见 `qqbot_relay/README.md`。

---

## 五、已知限制

1. **手机端还没有本地唤醒词**：现在靠"按住说话"这个手势表达意图（服务端按需唤醒）。
   真正的"随叫随到"需要原生 App + 本地 KWS（`docs/传输层规划.md` 阶段 4）。
2. **公网入口未接通**：局域网已可用；公网需要 NAS 反代或端口映射（阶段 5）。
3. **自签证书**：局域网 IP 签不出公信证书，手机需手动信任一次。
4. **回声**：网页端靠 `echoCancellation` + 播放时暂停采集；戴耳机可完全规避。
