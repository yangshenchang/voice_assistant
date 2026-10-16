/**
 * QQ 官方机器人「主动推送」轻量客户端（零依赖，只用 Node 内置 fetch）。
 *
 * 与 dsh-qqbot 插件的区别：插件是「入站消息 → agent → 回复」的前端传输，出站必须
 * 依赖入站消息上下文（cmdCtx/SessionRecord），因此**无法主动推送**；本模块绕开它,
 * 直接用平台 HTTP 接口发消息 —— 不需要 WebSocket，也不会和插件的网关连接冲突。
 *
 * 接口细节全部来自本机安装的 SDK（@tencent-connect/qqbot-nodejs）源码：
 *   token : POST https://bots.qq.com/app/getAppAccessToken   body={appId,clientSecret}
 *           响应 {access_token, expires_in}                   (token.js:8,145,150,181)
 *   发送  : POST https://api.sgroup.qq.com/v2/users/{openid}/messages
 *           header Authorization: `QQBot <token>`（不是 Bearer）(api-client.js:34)
 *           body   {markdown:{content}, msg_type:2}           (messages.js:196)
 *   主动消息不带 msg_id / msg_seq（带 msg_seq 的路径固定填 1）(messages.js:124,196)
 *   单条上限: TEXT_CHUNK_LIMIT = 5000，SDK 不会自动切分    (text-chunk.js:10)
 */

const TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken";
const API_BASE = "https://api.sgroup.qq.com";
const CHUNK_LIMIT = 4500; // 平台上限 ~5000，留余量（插件默认也是 4500）

let _cache = { appId: null, token: null, expiresAt: 0 };

/** 取 App Access Token（带缓存 + 提前刷新：剩余有效期不足 1/3 或不足 5 分钟就重取）。 */
export async function getAccessToken(appId, appSecret, { timeoutMs = 10000 } = {}) {
  const now = Date.now();
  if (_cache.appId === appId && _cache.token && now < _cache.expiresAt) return _cache.token;

  const res = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "User-Agent": "voice-assistant-qqpush/0.1" },
    body: JSON.stringify({ appId, clientSecret: appSecret }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const raw = await res.text();
  if (!res.ok) throw new Error(`获取 token 失败: HTTP ${res.status} ${raw.slice(0, 200)}`);
  const data = JSON.parse(raw);
  if (!data.access_token) throw new Error(`获取 token 失败: ${raw.slice(0, 200)}`);

  const ttl = (data.expires_in ?? 7200) * 1000;
  const ahead = Math.min(5 * 60 * 1000, ttl / 3); // 提前刷新窗口
  _cache = { appId, token: data.access_token, expiresAt: now + ttl - ahead };
  return _cache.token;
}

/** 按段落边界把长文切块（不切断 markdown 表格/代码块的基本形态）。 */
export function chunkText(text, limit = CHUNK_LIMIT) {
  const out = [];
  let buf = "";
  for (const para of String(text).split(/\n/)) {
    if ((buf + "\n" + para).length > limit && buf) {
      out.push(buf);
      buf = para;
    } else {
      buf = buf ? `${buf}\n${para}` : para;
    }
    while (buf.length > limit) {           // 单行本身超长 → 硬切
      out.push(buf.slice(0, limit));
      buf = buf.slice(limit);
    }
  }
  if (buf) out.push(buf);
  return out;
}

/**
 * 主动推送 markdown 消息给某个单聊用户。
 * @returns {Promise<{id?:string, timestamp?:string, chunks:number}>}
 */
export async function pushC2CMarkdown(openid, markdown, { appId, appSecret, markdownSupport = true } = {}) {
  if (!openid) throw new Error("缺少 openid（主动推送必须知道目标用户）");
  if (!markdown || !markdown.trim()) throw new Error("内容为空，QQ 会拒绝发送");

  const token = await getAccessToken(appId, appSecret);
  const chunks = chunkText(markdown);
  let last = {};
  for (const piece of chunks) {
    const body = markdownSupport
      ? { markdown: { content: piece }, msg_type: 2 }
      : { content: piece, msg_type: 0 };
    const res = await fetch(`${API_BASE}/v2/users/${openid}/messages`, {
      method: "POST",
      headers: {
        Authorization: `QQBot ${token}`,
        "Content-Type": "application/json",
        "User-Agent": "voice-assistant-qqpush/0.1",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    const raw = await res.text();
    if (!res.ok) {
      let code, message;
      try { const e = JSON.parse(raw); code = e.code ?? e.err_code; message = e.message; } catch { /* 非 JSON */ }
      throw new Error(`QQ 推送失败: HTTP ${res.status} code=${code} msg=${message ?? raw.slice(0, 200)}`);
    }
    try { last = JSON.parse(raw); } catch { /* 忽略 */ }
  }
  return { id: last.id, timestamp: last.timestamp, chunks: chunks.length };
}
