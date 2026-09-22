# Hermes 语音助手（混合架构）实施方案

> 版本: v1.0 ｜ 日期: 2026-08-21 ｜ 目标环境: 飞牛NAS (fnOS) ｜ 状态: 核心链路已实测验证

---

## 一、背景与目标

### 1.1 我们要做什么

构建一个**本地能力强大且回复迅速的语音助手**：

- 用户唤醒语音助手并说出问题（麦克风）
- 助手实时语音回复（扬声器）
- 涉及本地操作（命令、文件、网络、矿山 MCP 系统）时，自动调用本地能力执行
- 复杂任务（分析、汇总、写报告）交给大模型完整处理

### 1.2 为什么选这套方案

最初尝试把阿里 `qwen3.5-omni-flash-realtime` 直接配置为 Hermes 主模型，**失败**。根因分析：

| 尝试 | 结果 | 原因 |
|---|---|---|
| HTTP chat completions 调用 realtime 模型 | ❌ 无响应 | 该模型只走 WebSocket 实时协议，HTTP 调用仅返回空壳 `{"status_message":"Success."}` |
| 配置为 Hermes 主模型 | ❌ 无法连接 | Hermes 主通道是 HTTP，协议不匹配 |

关键认知：**realtime 模型不是"不能用于 Hermes 架构"，而是"它应该当大脑，Hermes 当手"。** 两者协议不同、职责不同，正好分工协作。

### 1.3 最终架构形态（已实测）

```
┌────────────────────────────────────────────────────────────┐
│  客户端层（NAS 上，同机）                                     │
│  麦克风采集 / 唤醒词 / 扬声器播放                             │
└───────────────┬───────────────────────┬────────────────────┘
                │ 音频流 (16kHz PCM)      │ 播放 (24kHz PCM)
                ▼                        ▲
┌────────────────────────────────────────────────────────────┐
│  桥接层 realtime_bridge.py（新写的核心，唯一的新代码）         │
│  A. WebSocket 直连阿里 realtime 模型（云端）                 │
│     - 上行: 麦克风 → input_audio_buffer.append              │
│     - 下行: response.audio.delta → 扬声器                   │
│  B. 工具注册表（session.update 注册 6 个工具）                │
│     - 5 个轻量工具: terminal/read_file/write_file/          │
│       search_files/web_search                               │
│     - 1 个路由工具: delegate_to_hermes（复杂任务入口）        │
└───────┬───────────────────────────────┬────────────────────┘
        │ function_call（轻量）          │ delegate（复杂任务）
        ▼                               ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ Hermes 工具执行器     │   │ Hermes 完整 agent 子进程       │
│ handle_function_call │   │ hermes chat --provider X -m Y │
│ （同机 import，零网络）│   │ （带 memory/上下文/多轮/技能）  │
└──────────────────────┘   └──────────────────────────────┘
```

### 1.4 为什么用混合架构（而非单一 agent）

| 能力 | 只用 realtime 模型 | 只用 Hermes agent | 混合（本方案） |
|---|---|---|---|
| 实时流式语音 | ✅ | ❌ 分段式 | ✅ |
| 本地工具执行 | ✅（function calling） | ✅ | ✅ |
| 长任务多轮推理 | ❌ 上下文有限 | ✅ | ✅（delegate） |
| Hermes memory/技能 | ❌ | ✅ | ✅（delegate） |
| 响应速度 | 最快 | 慢 | 快 + 复杂任务稍慢 |
| 成本 | 音频计费 | token 计费 | 各用各的，互不干扰 |

**核心洞察：谁当大脑，谁拥有记忆/上下文/多轮。**
- 实时语音对话历史 → realtime 模型自己记（WS 会话内）
- 复杂任务的长期记忆/工具链 → delegate 给 Hermes agent

---

## 二、模型配置（语音与推理解耦）

### 2.1 实时语音模型（固定）

```
模型: qwen3.5-omni-flash-realtime
连接: WebSocket wss://{host}/api-ws/v1/realtime?model=qwen3.5-omni-flash-realtime
认证: Authorization: Bearer $DASHSCOPE_API_KEY
模态: ["text", "audio"]（模型自带 ASR + VAD + TTS，无需外部语音组件）
```

### 2.2 复杂任务推理模型（可独立切换）

在桥接层配置文件中通过两个常量指定（已验证可切换）：

```
DELEGATE_PROVIDER = "deepseek"        # 复杂任务推理的 provider
DELEGATE_MODEL    = "deepseek-v4-flash"

# 已验证的备选组合（NAS 上改这两个常量即可切换）:
#   deepseek / deepseek-v4-flash   → 快（当前默认）
#   deepseek / deepseek-v4-pro     → 更强推理
#   alibaba  / qwen3.8-max         → 阿里旗舰文本（走专属实例，实测 ~4s）
#   alibaba  / qwen3.7-max         → 备选
```

**重要踩坑**：`hermes chat -m alibaba/qwen3.8-max` 斜杠格式**不会切换 provider**（实测会把 `alibaba/qwen3.8-max` 整个当模型名发给 deepseek 端点 → HTTP 400）。必须分开传：

```bash
hermes chat -q "任务" --provider alibaba -m qwen3.8-max
```

### 2.3 环境变量（~/.hermes/.env）

```
DASHSCOPE_API_KEY=<你的阿里 API Key>
DASHSCOPE_BASE_URL=https://llm-ney31dctilhqc6ku.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

（delegate 用 deepseek 时还需 `DEEPSEEK_API_KEY`，或在 Hermes config.yaml 配置对应 provider）

---

## 三、混合路由逻辑（核心）

### 3.1 路由决策者 = realtime 模型自己

不是代码判断，而是**靠工具描述引导模型自主决策**。桥接层注册一个特殊工具：

```
工具名: delegate_to_hermes
参数:   { "task": "完整任务描述" }
描述:   "当任务需要多步推理、长期记忆、生成文档/报告、复杂工具链、
        或用户要求'帮我处理/分析/汇总'时使用。任务描述必须完整自包含
        （执行端看不到当前对话）。简单问答不要调用。"
```

### 3.2 三种场景的走向（均已实测）

| 用户请求 | 模型行为 | 实测结果 |
|---|---|---|
| "1+1等于几" | 直接回答，零工具 | ✅ "1加1等于2" |
| "用terminal执行date命令" | 调轻量工具 → 执行 → 回传 → 播报 | ✅ 返回当前时间 |
| "写一首四句诗" | 判定复杂 → 调 delegate_to_hermes | ✅ Hermes agent 写了首诗并播报 |

### 3.3 上下文分工（最容易混淆的点）

- **语音对话历史** → realtime 模型（WS 会话内自动累积）
- **实质任务上下文** → delegate 时任务描述必须**完整自包含**（Hermes 看不到当前对话）
- 可选进阶：delegate 复用固定 `--session` ID，让 Hermes 端也有长期记忆

---

## 四、NAS 部署步骤

### 4.1 前置准备

1. **安装 Python 3.11+**（fnOS 一般自带；无则用 uv 安装）
2. **安装 Hermes**（按官方 install.sh，或手动部署源码）
3. **配置阿里凭据**（见 2.3，写入 NAS 上 Hermes 的 `~/.hermes/.env`）
4. **安装桥接层依赖**：

```bash
uv pip install --python /path/to/hermes/venv/bin/python websockets
# 音频采集需要（NAS 音频设备可用时）:
uv pip install --python /path/to/hermes/venv/bin/python pyaudio
```

### 4.2 部署桥接层

1. 将 `realtime_bridge.py` 拷贝到 NAS（如 `~/scripts/realtime_bridge.py`）
2. 修改文件头部配置：

```python
# 部署环境差异只需要改这两处:
HERMES_DIR = "/path/to/hermes-agent"   # NAS 上 Hermes 源码目录
DELEGATE_PROVIDER = "deepseek"         # 复杂任务推理 provider
DELEGATE_MODEL = "deepseek-v4-flash"   # 复杂任务推理模型
```

3. **连通性自检**（文字模式，不接音频先验证）：

```bash
python realtime_bridge.py
# 输入: 用terminal执行 date 命令告诉我现在时间
# 应看到: 模型自主调 terminal → 返回时间 → 语音风格播报
```

### 4.3 接入客户端（你的麦克风+唤醒+音响逻辑）

桥接层已预留两个音频接口，客户端只需对接：

```python
# 1. 模型 → 扬声器（注册播放回调）
bridge.on_audio_output = lambda pcm_b64: 你的播放函数(pcm_b64)

# 2. 麦克风 → 模型（采集循环里持续推流）
await bridge.feed_audio(pcm_bytes)   # 16kHz 16bit PCM

# 3. 唤醒后建立连接
await bridge.connect(api_key, base_url)
```

**VAD（检测人说完话）由模型端 server_vad 自动处理，客户端无需实现。**

### 4.4 启动方式

- 前台调试: `python realtime_bridge.py`
- 常驻运行: 建议用 systemd 服务或 supervisor，开机自启
- 与唤醒词进程配合: 唤醒后拉起桥接层，超时静默后退出

---

## 五、验证清单（NAS 部署后逐项确认）

| # | 验证项 | 预期 |
|---|---|---|
| 1 | curl 模型列表 | 200，含 qwen3.5-omni-flash-realtime |
| 2 | 文字模式连接 | 打印"会话就绪, 注册 6 个工具" |
| 3 | 普通问答 | 模型直接语音/文字回复 |
| 4 | 轻量工具 | 模型自主调 terminal，返回真实执行结果 |
| 5 | delegate 复杂任务 | 模型自主调 delegate_to_hermes，Hermes 完整处理并播报 |
| 6 | 模型切换 | 改 DELEGATE_PROVIDER/MODEL 后 delegate 使用新模型 |
| 7 | 音频链路 | 麦克风说话 → 模型理解 → 语音回复播放 |

---

## 六、协议要点与踩坑记录（调试必备）

### 6.1 阿里 Qwen-Omni-Realtime 协议要点

1. **只走 WebSocket**，HTTP chat completions 仅返回空壳状态
2. 端点: `wss://{host}/api-ws/v1/realtime?model=xxx` —— **model 必须是 URL 查询参数**（放 body 会 1009 报错）
3. **不支持 `tool_choice` 和 `parallel_tool_calls`**（带上模型静默不响应）
4. `response.create` 应**裸发**（`{"type":"response.create"}`），带多余参数会卡住
5. 工具声明用**扁平格式**：`{"type":"function","name":...,"description":...,"parameters":...}`
6. **function_call_output 必须等上一轮 response.done 之后发送**（提前发模型会重复调用工具）
7. 模态 `["text","audio"]` 时文字走 `response.audio_transcript.delta`（不是 `response.text.delta`）；纯文本模态才走 text.delta
8. 工具结果回传前截断（~6000 字符），防止撑爆实时会话上下文

### 6.2 Hermes 侧踩坑

1. `hermes chat -m provider/model` 斜杠格式不切换 provider → 用 `--provider X -m Y` 分开传
2. `handle_function_call(name, args)` 可直接从独立脚本调用（同 venv 内），返回 JSON 字符串
3. `get_tool_definitions(quiet_mode=True)` 获取工具 schema（注意是 `{"type":"function","function":{...}}` 嵌套格式，需转扁平）

### 6.3 事件流速查（一轮完整工具调用）

```
客户端: session.update（注册工具）
服务端: session.created → session.updated
客户端: conversation.item.create(用户消息) → response.create
服务端: response.created → response.output_item.added
        → response.function_call_arguments.done (call_id, name, args)
        → response.done
客户端: conversation.item.create(function_call_output) → response.create
服务端: response.audio_transcript.delta / response.audio.delta → response.done
```

---

## 七、后续演进路线（本次不实现，设计已预留）

1. **音频接通**: pyaudio 采集/播放，客户端对接两个回调接口
2. **唤醒词**: 复用你现有唤醒逻辑，唤醒后建立 WS 连接
3. **MCP 接入**: 矿山自研 MCP 配进 Hermes（`hermes mcp add`），桥接层 LIGHT_TOOLS 加上 MCP 工具名即可让语音助手调用
4. **任务分级**: delegate 工具增加可选 `model` 参数，让模型按任务复杂度选模型
5. **长期记忆**: delegate 复用固定 `--session` ID，Hermes 端跨次对话有连续上下文
6. **打断处理**: 监听 `response.cancel` / 新语音输入时取消当前回复

---

## 八、文件清单

| 文件 | 位置（当前开发机） | 说明 |
|---|---|---|
| 桥接层 | `/home/zj/scripts/realtime_bridge.py` | 混合模式核心，部署时拷到 NAS |
| 音频测试 | `/home/zj/scripts/test_qwen_realtime.py` | realtime 模型文字交互测试 |
| 技能存档 | `aliyun-omni-realtime-voice-assistant` | 协议要点+踩坑，Hermes 技能 |
| Hermes 别名 | config.yaml `model.aliases.ali-omni` | 阿里实例文本模型（qwen3.5-omni-flash） |

---

## 九、风险与注意事项

1. **NAS 音频设备**：fnOS 上麦克风/扬声器访问依赖 PulseAudio/ALSA，需确认 NAS 硬件支持并安装对应驱动；若 NAS 无麦克风，客户端可改为手机/其他设备远程接入（桥接层音频接口不变，网络化是后续工作）
2. **实时模型成本**：realtime 按音频时长计费，长对话需关注费用
3. **delegate 耗时**：复杂任务几秒~几十秒，模型调用工具前会先说"请稍等"，交互体验可接受；超时上限 300s
4. **安全**：NAS 上运行的 agent 具备完整本地权限（terminal 等），建议仅在内网使用，唤醒词鉴权
