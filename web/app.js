/**
 * 手机端「通话模式」客户端。
 *
 * 形态：点「接通」→ 自动连服务端 + 持续推流 → 直接说话 → 听完接着说。
 * 不再需要按住/松开按钮：**判句完全交给服务端 VAD**（与本机麦克风路径同一套逻辑）。
 *
 * 三条关键设计：
 *
 * 1. **持续推流，不做本地判句**。服务端的 VAD（振幅阈值 + 静音 0.5s 判句）本来就是
 *    这条链路的判句者；客户端只要一直喂音频，行为就与本机麦克风完全一致，
 *    时延特性也能直接沿用既有基线。
 *
 * 2. **回声抑制（默认开，半双工）**。麦克风一直开着，助手的回答会被自己收进去 →
 *    自问自答。默认做法是"助手出声时暂停采集"（本机 `cancel_echo` 的对等物）。
 *    代价是这段时间不能打断 —— 与现在的体验一致，但胜在**可靠**。
 *    戴耳机时可勾选「免提全双工」放宽为只靠 `echoCancellation`。
 *
 * 3. **状态以服务端为准**。`state` 是"现在能不能说话/正在发生什么"的唯一权威；
 *    本地只在服务端失联时用看门狗兜底（否则界面会假死）。
 */

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;              // 32ms
const BIN_UPLINK = 0x01;
const BIN_DOWNLINK = 0x02;
const PROTOCOL_VERSION = 1;

/** 播放缓冲余量（秒）：太小断续，太大增延迟。 */
const PLAY_LEAD = 0.12;
/** 本地看门狗：服务端状态超时未更新则解锁，避免界面假死。 */
const MAX_BUSY_MS = 25000;
/** 回声抑制尾部：最后一块下行音频到达后再静音这么久（覆盖扬声器余音）。 */
const ECHO_TAIL_MS = 350;
/** 断线自动重连的退避（毫秒）。 */
const RECONNECT_DELAY_MS = 1500;

const $ = (id) => document.getElementById(id);
const logEl = $("log");
const TOKEN_KEY = "xiaoniu.token";
const AUTO_KEY = "xiaoniu.auto";
const DUPLEX_KEY = "xiaoniu.duplex";

function log(text, cls = "sys") {
  const line = document.createElement("div");
  line.className = cls;
  line.textContent = text;
  logEl.appendChild(line);
  // 日志过长时裁剪，避免长时间通话把内存堆满
  while (logEl.childElementCount > 400) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

const state = {
  ws: null,
  sessionId: null,
  inCall: false,          // 通话中（已接通）
  connecting: false,
  serverIdle: true,       // 服务端说"能接新的一轮"
  serverSpeaking: false,  // 服务端正在送音频
  serverRecording: false,
  busySince: 0,
  callStart: 0,
  uplinkFrames: 0,
  dropped: 0,
  rtt: null,
  lastDownlinkAt: 0,      // 最后一次收到下行音频的时刻（回声抑制用）
  echoMuted: false,       // 当前是否因回声抑制而暂停上行
  duplex: false,          // 全双工（不做回声抑制）
  audioCtx: null,
  captureNode: null,
  micStream: null,
  workletReady: false,
  nextPlayTime: 0,
  activeSources: new Set(),
  turnStart: 0,
  turnFirstAudio: null,
  manualHangup: false,
};

// ---------------------------------------------------------------------------
// 状态展示
// ---------------------------------------------------------------------------

function setVisual(kind, text, hint) {
  const avatar = $("avatar");
  avatar.className = "avatar " + kind;
  const st = $("state");
  st.className = "state " + kind;
  st.textContent = text;
  if (hint !== undefined) $("hint").textContent = hint;
}

function refreshVisual() {
  if (!state.inCall) { setVisual("idle", "未接通", "点下方按钮接通，然后直接说话"); return; }
  if (state.serverSpeaking) return setVisual("speaking", "助手正在说…", "听完直接接着说");
  if (!state.serverIdle) return setVisual("thinking", "正在处理…", "稍等，处理完可以直接说下一句");
  if (state.echoMuted) return setVisual("thinking", "准备中…", "马上就好");
  return setVisual("listening", "正在听…", "直接说话，停顿一下我就会回答");
}

function updateCallButton() {
  const btn = $("call");
  if (state.connecting) { btn.disabled = true; btn.className = "busy"; btn.textContent = "接通中…"; return; }
  btn.disabled = false;
  if (state.inCall) { btn.className = "on"; btn.textContent = "挂断"; }
  else { btn.className = ""; btn.textContent = "接通"; }
}

function updateEchoBadge() {
  const el = $("echo");
  if (state.duplex) { el.textContent = "关（全双工）"; el.className = "warn"; }
  else { el.textContent = state.echoMuted ? "抑制中" : "开"; el.className = "ok"; }
}

// ---------------------------------------------------------------------------
// 连接
// ---------------------------------------------------------------------------

function savePrefs() {
  try {
    const t = $("token").value.trim();
    if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY);
    localStorage.setItem(AUTO_KEY, $("auto").checked ? "1" : "0");
    localStorage.setItem(DUPLEX_KEY, $("duplex").checked ? "1" : "0");
  } catch { /* 隐私模式下不可用，忽略 */ }
}

function connect() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) return;
  if (state.connecting) return;
  state.connecting = true;
  state.manualHangup = false;
  updateCallButton();

  const token = $("token").value.trim();
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws`;
  log(`连接 ${url} …`);
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = () => {
    // ⚠️ 握手必须**立刻**发、不做任何 await：手机上"申请麦克风权限"会弹窗停留数秒，
    // 服务端 10s 收不到 hello 就断开（实测踩过：界面显示"连不上"，其实是握手被 await 拖死）。
    ws.send(JSON.stringify({
      type: "hello", version: PROTOCOL_VERSION, token,
      sampleRate: TARGET_RATE, frameSamples: FRAME_SAMPLES,
    }));
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") onControl(ev.data);
    else onAudio(new Uint8Array(ev.data));
  };
  ws.onclose = () => {
    const wasInCall = state.inCall;
    state.connecting = false;
    state.inCall = false;
    state.ws = null;
    stopPlayback();
    stopCapture();
    updateCallButton();
    refreshVisual();
    log("— 已断开 —", "warn");
    // 通话中被断开（网络抖动/服务端重启）→ 自动重连，电话不该因为一次抖动就结束
    if (wasInCall && !state.manualHangup) {
      log(`将在 ${RECONNECT_DELAY_MS}ms 后自动重连…`, "warn");
      setTimeout(() => { if (!state.manualHangup) startCall(); }, RECONNECT_DELAY_MS);
    }
  };
  ws.onerror = () => log("WebSocket 错误（详情看服务端日志）", "err");
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function onControl(raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch { return; }
  switch (msg.type) {
    case "ready":
      state.connecting = false;
      state.inCall = true;
      state.sessionId = msg.sessionId;
      if (msg.protocol !== PROTOCOL_VERSION) {
        log(`⚠ 协议版本不一致：服务端 ${msg.protocol} / 客户端 ${PROTOCOL_VERSION}`, "err");
      }
      state.callStart = Date.now();
      updateCallButton();
      log(`已接通 · 会话 ${msg.sessionId} · ${msg.sampleRate}Hz`, "sys");
      beginStreaming();     // 接通后**立刻开始持续推流**
      pingLoop();
      break;
    case "state":
      applyServerState(msg);
      break;
    case "stt":
      state.turnFirstAudio = null;
      log(`我: ${msg.text}`, "me");
      break;
    case "text":
      log(`小牛: ${msg.text}`, "ai");
      break;
    case "announce":
      log(`小牛（播报）: ${msg.text}`, "ai");
      break;
    case "turn":
      if (msg.phase === "start") { state.turnStart = performance.now(); $("turn").textContent = "…"; }
      else if (msg.outcome && msg.outcome !== "ok") {
        // 失败类结果也显示出来，user 才知道发生了什么（不然就是"没反应"）
        $("turn").textContent = msg.outcome;
      }
      break;
    case "notice":
      log(`· ${msg.message || msg.code}`, msg.code === "turn_error" ? "err" : "warn");
      break;
    case "pong":
      if (typeof msg.t === "number") {
        state.rtt = Math.round(performance.now() - msg.t);
        $("rtt").textContent = `${state.rtt} ms`;
      }
      break;
    case "error":
      log(`错误[${msg.code}] ${msg.message}`, "err");
      break;
    case "bye":
      log(`服务端要求断开：${msg.reason}`, "warn");
      break;
  }
}

function applyServerState(msg) {
  state.serverIdle = !!msg.idle;
  state.serverSpeaking = !!msg.speaking;
  state.serverRecording = !!msg.recording;
  if (msg.idle && !msg.speaking) state.busySince = 0;
  else if (!state.busySince) state.busySince = Date.now();
  refreshVisual();
}

function pingLoop() {
  if (!state.inCall) return;
  send({ type: "ping", t: performance.now() });
  setTimeout(pingLoop, 3000);
}

// ---------------------------------------------------------------------------
// 通话控制
// ---------------------------------------------------------------------------

async function startCall() {
  if (state.inCall || state.connecting) return;
  savePrefs();
  updateCallButton();
  try {
    // 在**用户手势**里申请麦克风：既符合浏览器要求，权限弹窗也只会出现一次
    await ensureMic();
  } catch (ex) {
    state.connecting = false;
    updateCallButton();
    setVisual("idle", "麦克风不可用", String(ex.message || ex));
    log(`拿不到麦克风：${ex.message}`, "err");
    return;
  }
  connect();
}

function hangup() {
  state.manualHangup = true;
  stopCapture();
  stopPlayback();
  send({ type: "end" });
  try { state.ws && state.ws.close(); } catch { /* ignore */ }
  state.inCall = false;
  state.connecting = false;
  updateCallButton();
  refreshVisual();
  log("通话结束", "sys");
}

function beginStreaming() {
  if (!state.captureNode) return;
  state.captureNode.port.postMessage({ type: "start" });
  send({ type: "start" });
  setEchoMuted(false);
  refreshVisual();
}

function setEchoMuted(muted) {
  if (state.duplex) muted = false;      // 全双工：不做回声抑制
  if (state.echoMuted === muted) return;
  state.echoMuted = muted;
  if (state.captureNode) {
    // 不做"停止采集"，只让 worklet 丢弃样本：避免反复 start/stop 造成音频线程状态抖动
    state.captureNode.port.postMessage({ type: muted ? "pause" : "resume" });
  }
  updateEchoBadge();
  if (state.inCall) refreshVisual();
}

/** 回声抑制的判定：正在收到/刚收到下行音频 → 静音上行。 */
function scheduleEchoGuard() {
  if (state.duplex) return;
  setEchoMuted(true);
  clearTimeout(state._echoTimer);
  state._echoTimer = setTimeout(() => {
    const sinceLast = performance.now() - state.lastDownlinkAt;
    if (sinceLast >= ECHO_TAIL_MS) setEchoMuted(false);
    else scheduleEchoGuard();     // 仍在放音，续期
  }, ECHO_TAIL_MS);
}

// ---------------------------------------------------------------------------
// 音频：采集
// ---------------------------------------------------------------------------

function isSecureContextOk() {
  return window.isSecureContext === true
      || location.hostname === "localhost" || location.hostname === "127.0.0.1";
}

async function ensureMic() {
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (state.audioCtx.state === "suspended") await state.audioCtx.resume();
  if (!state.workletReady) {
    await state.audioCtx.audioWorklet.addModule("./capture-worklet.js");
    state.workletReady = true;
  }
  if (state.micStream) return;

  // 浏览器只在**安全上下文**里给麦克风：http://局域网IP 会被直接拒绝（http://localhost 例外）
  if (!isSecureContextOk()) {
    throw new Error(
      "当前页面不是安全上下文，浏览器会拒绝麦克风。请用 https 访问，" +
      `或在 Chrome 把 http://${location.host} 加入 ` +
      "chrome://flags/#unsafely-treat-insecure-origin-as-secure 后重启浏览器。");
  }
  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      // 通话形态下这条是**第一道回声防线**（第二道是下面的上行静音）
      echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      channelCount: 1,
    },
  });
  const src = state.audioCtx.createMediaStreamSource(state.micStream);
  state.captureNode = new AudioWorkletNode(state.audioCtx, "capture-processor", {
    numberOfInputs: 1, numberOfOutputs: 0,
    processorOptions: { targetRate: TARGET_RATE, frameSamples: FRAME_SAMPLES },
  });
  state.captureNode.port.onmessage = (e) => {
    if (!state.inCall || state.echoMuted) return;   // 回声抑制期间直接丢弃
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    const pcm = new Uint8Array(e.data);
    const frame = new Uint8Array(4 + pcm.length);
    frame[0] = BIN_UPLINK; frame[1] = 0;
    const seq = state.uplinkFrames & 0xffff;
    frame[2] = seq & 0xff; frame[3] = (seq >> 8) & 0xff;
    frame.set(pcm, 4);
    state.ws.send(frame.buffer);
    state.uplinkFrames++;
    $("up").textContent = state.uplinkFrames;
  };
  src.connect(state.captureNode);
  state.echoMuted = false;
  updateEchoBadge();
}

function stopCapture() {
  if (state.captureNode) state.captureNode.port.postMessage({ type: "stop" });
}

// ---------------------------------------------------------------------------
// 音频：播放
// ---------------------------------------------------------------------------

function onAudio(frame) {
  if (frame.length < 4) return;
  if (frame[0] !== BIN_DOWNLINK) return;
  const pcm = frame.subarray(4);
  if (pcm.length < 2) return;

  state.lastDownlinkAt = performance.now();
  scheduleEchoGuard();                    // 收到下行音频 → 暂停上行，防自问自答

  if (state.turnFirstAudio === null && state.turnStart) {
    state.turnFirstAudio = performance.now() - state.turnStart;
    $("turn").textContent = `${Math.round(state.turnFirstAudio)} ms`;
  }
  const ctx = state.audioCtx;
  if (!ctx) return;
  const n = pcm.length >> 1;
  const buf = ctx.createBuffer(1, n, TARGET_RATE);
  const ch = buf.getChannelData(0);
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  for (let i = 0; i < n; i++) ch[i] = view.getInt16(i * 2, true) / 32768;

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  const when = Math.max(now + PLAY_LEAD, state.nextPlayTime || 0);
  src.start(when);
  state.nextPlayTime = when + buf.duration;
  state.activeSources.add(src);
  src.onended = () => state.activeSources.delete(src);
}

function stopPlayback() {
  for (const s of state.activeSources) { try { s.stop(); } catch { /* 已结束 */ } }
  state.activeSources.clear();
  state.nextPlayTime = 0;
}

// ---------------------------------------------------------------------------
// 看门狗与计时
// ---------------------------------------------------------------------------

setInterval(() => {
  // 提示超时兜底：服务端忙太久就本地解锁（否则界面永远停在"正在处理…"）
  if (state.inCall && (!state.serverIdle || state.serverSpeaking) && state.busySince) {
    if (Date.now() - state.busySince > MAX_BUSY_MS) {
      log("服务端状态超时未更新，已本地解锁（若反复出现请把服务端日志发我）", "warn");
      state.serverIdle = true; state.serverSpeaking = false; state.busySince = 0;
      refreshVisual();
    }
  }
  // 通话时长
  if (state.inCall && state.callStart) {
    const s = Math.floor((Date.now() - state.callStart) / 1000);
    $("dur").textContent =
      String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }
}, 1000);

// ---------------------------------------------------------------------------
// 事件绑定
// ---------------------------------------------------------------------------

$("call").addEventListener("click", () => { state.inCall ? hangup() : startCall(); });

$("duplex").addEventListener("change", (e) => {
  state.duplex = e.target.checked;
  savePrefs();
  if (state.duplex) { log("已切到免提全双工：不做回声抑制，可能自问自答（建议戴耳机）", "warn"); setEchoMuted(false); }
  else log("已切回半双工：助手说话期间会暂停采集（更可靠）", "sys");
  updateEchoBadge();
});

$("token").addEventListener("keydown", (e) => { if (e.key === "Enter" && !state.inCall) startCall(); });

// 页面切后台/锁屏：浏览器会限制麦克风，通话实际会中断 —— 明确告诉用户，别让他以为坏了
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.inCall) {
    log("页面进入后台：手机浏览器会限制麦克风，通话可能中断（回到前台会自动恢复）", "warn");
  } else if (!document.hidden && state.inCall && state.ws && state.ws.readyState === WebSocket.OPEN) {
    beginStreaming();
  }
});

// 关页面前尽量礼貌挂断
window.addEventListener("pagehide", () => { if (state.inCall) { send({ type: "end" }); } });

// 恢复偏好 + 尝试自动接通
try {
  const saved = localStorage.getItem(TOKEN_KEY);
  if (saved) { $("token").value = saved; }
  const auto = localStorage.getItem(AUTO_KEY);
  if (auto !== null) $("auto").checked = auto === "1";
  const dx = localStorage.getItem(DUPLEX_KEY);
  if (dx !== null) { $("duplex").checked = dx === "1"; state.duplex = dx === "1"; }
} catch { /* ignore */ }
updateEchoBadge();
updateCallButton();
refreshVisual();

// 页面加载后自动接通（"打开即通话"）。浏览器要求麦克风必须由用户手势触发，
// 若被拦下就退回手动点「接通」——所以提示文案要说清这一点。
if ($("auto").checked) {
  log("自动接通中…（若浏览器拦截了麦克风，请点一下「接通」）", "sys");
  startCall();
}
log("提示：接通后直接说话，停顿一下我就回答；说完等我说完再接着说。", "sys");
