/**
 * dsh-qqbot-relay —— 语音助手「复杂路径」的常驻委托中继。
 *
 * 挂在 qqbot profile 内（该进程为 QQ 机器人常驻，冷启动摊薄为零），做三件事：
 *   ① 控制面: 本地 HTTP，语音助手用它提交任务 / 查询进度 / 取结果
 *   ② 驱动  : 创建一个**工具受限**的 dsh agent，把任务喂给它，实时消费它的会话事件
 *   ③ 投递  : 完整报告主动推送到用户 QQ（走平台 HTTP 接口，见 qq_push.js）
 *
 * API 全部依据本机安装的 dsh 0.1.2-rc.1 类型声明：
 *   ctx.agents.create({sessionId, meta, agentOptions, setup})  dsh-agent/lib/types/index.d.ts:283
 *   setup(agentCtx) → agentCtx.tools.restrict({allow:[...]})   dsh-tools/lib/types/index.d.ts:610 / :471
 *   agent.followup(createUserMessage({content, source}))       dsh-agent/.../runtime-types.d.ts:118
 *   agent.whenIdle() / agent.cancel({kind})                    runtime-types.d.ts:90 / :83
 *   ctx.on('session/event', (session, event) => …)             dsh-session/lib/types/index.d.ts:64
 *   agent.session.snapshotEvents(fromSeq) / seq                dsh-session/lib/types/index.d.ts:184 / :197
 *   AgentHandle.dispose()                                      dsh-agent/lib/types/index.d.ts:150-153
 * 事件词汇表 session 事件: turn/start、turn/end、assistant/message、tool/call、tool/result
 *                          (dsh-session/lib/types/types.d.ts:242/253/291/303/321)
 */

import http from 'node:http'
import { randomUUID } from 'node:crypto'

import Schema from '@deepseek-ai/schemastery'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'

import { pushC2CMarkdown } from './qq_push.js'

export const name = 'qqbot-relay'
export const inject = ['agents']

export const Config = Schema.object({
  host: Schema.string().default('127.0.0.1'),
  port: Schema.number().default(8765),
  /** dsh agent 的工作目录（沙箱边界） */
  cwd: Schema.string().default(process.cwd()),
  /** 留空则继承宿主默认模型路由（注意：宿主不一定有默认路由，实测裸 profile 会报
   *  "has no provider/model"，所以这里给出与 settings.yaml 一致的显式默认值） */
  provider: Schema.string().default('deepseek-official'),
  model: Schema.string().default('deepseek-flash'),
  /** 工具白名单：allow = 只保留这些，其余对 agent 不可见且拒绝执行 */
  allowedTools: Schema.array(Schema.string()).default([
    'web_search', 'web_fetch', 'read', 'glob', 'grep',
  ]),
  /** QQ 主动推送（openid 留空则只记账、不推送）。凭据留空时回退到环境变量，
   *  便于后续把明文 secret 从 profile 配置里挪出去。 */
  qqAppId: Schema.string().default(''),
  qqAppSecret: Schema.string().default(''),
  openid: Schema.string().default(''),
  pushFullResult: Schema.boolean().default(true),
  /** 单任务超时 */
  taskTimeoutMs: Schema.number().default(600000),
  debug: Schema.boolean().default(true),
})

/** 让模型把「口语结论」和「完整报告」分成两段输出，一次调用得到两个产物。 */
const DETAIL_SEP = '===详情==='
const TASK_PROMPT = (task) => `语音助手转来的任务：${task}

请完成任务。回复格式必须严格遵守两段：
第一段（会被语音合成朗读，务必能听懂）：1-2 句中文口语结论，直接给答案。
  - 禁止出现：文件名、扩展名（如 .py/.md）、路径、英文单词、markdown 符号、表格、代码、链接。
  - 不要说"根据搜索""详情见下文"这类废话，也不要把数字念成符号（写"百分之一点二"而不是"1.2%"）。
然后单独一行输出 ${DETAIL_SEP}
第二段：完整详细结果（可以有表格/列表/数据/链接），给人阅读用。
若任务很简单、无需详情，可以省略第二段和分隔线。`

/** 极简的最终回复提取：最后一个非空 assistant 文本块（含 text-delta 兜底）。 */
function extractFinalText(events) {
  let last = ''
  let streamed = ''
  for (const e of events) {
    if (e.type === 'assistant/message') {
      const blocks = e.data?.message?.content ?? []
      const text = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('').trim()
      if (text) last = text
    } else if (e.type === 'assistant/chunk') {
      const c = e.data?.chunk
      if (c?.type === 'text-delta' && c.text) streamed += c.text
    }
  }
  return last || streamed.trim()
}

/** 拆出「口语结论」与「完整报告」。没有分隔线时用前两句兜底当结论。 */
function splitResult(text) {
  if (!text) return { headline: '', full: '' }
  const idx = text.indexOf(DETAIL_SEP)
  if (idx >= 0) {
    return {
      headline: text.slice(0, idx).trim(),
      full: text.slice(idx + DETAIL_SEP.length).trim() || text.trim(),
    }
  }
  const sentences = text.split(/(?<=[。！？!?])/).filter((s) => s.trim())
  const headline = sentences.slice(0, 2).join('').trim() || text.slice(0, 120)
  return { headline, full: text.trim() }
}

export async function apply(ctx, config) {
  const log = (msg) => (config.debug ? console.log(`[qqbot-relay] ${msg}`) : undefined)

  // 凭据：配置优先，缺省回退环境变量（避免在 profile 配置里再存一份明文 secret）
  const qqAppId = config.qqAppId || process.env.QQBOT_APPID || ''
  const qqAppSecret = config.qqAppSecret || process.env.QQBOT_SECRET || ''
  const openid = config.openid || process.env.QQBOT_OPENID || ''

  /** taskId -> 任务记录 */
  const tasks = new Map()
  const queue = []
  let current = null
  let handle = null
  let agent = null
  let offEvent = null
  let offError = null
  let lastText = ''
  let lastTurnKind = null
  let lastTurnDetail = ''

  // ── 进度：订阅会话事件（插件 ctx 未打 scope tag ⇒ 收到所有 session，自行过滤） ──
  function attachProgress(a) {
    offEvent = ctx.on('session/event', (session, event) => {
      if (session !== a.session) return
      const rec = current
      const push = (text) => {
        if (!rec) return
        rec.progress.push({ at: Date.now(), text })
        if (rec.progress.length > 60) rec.progress.splice(0, rec.progress.length - 60)
        log(`task ${rec.id} 进度: ${text}`)
      }
      switch (event.type) {
        case 'turn/start':
          break
        case 'tool/call': {
          const nm = event.data?.name
          const args = String(event.data?.arguments ?? '')
          push(`正在使用 ${nm}${nm === 'web_search' || nm === 'web_fetch' ? '' : args ? ' ' + args.slice(0, 60) : ''}`)
          break
        }
        case 'tool/result':
          if (event.data?.error) push(`工具失败: ${event.data.error.name ?? ''}`)
          break
        case 'assistant/message': {
          const blocks = event.data?.message?.content ?? []
          const text = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('').trim()
          if (text) lastText = text
          break
        }
        case 'turn/end': {
          const reason = event.data?.reason ?? {}
          lastTurnKind = reason.kind ?? null
          if (lastTurnKind && lastTurnKind !== 'completed') {
            lastTurnDetail = JSON.stringify(reason).slice(0, 400)
            push(`本轮结束: ${lastTurnKind} ${lastTurnDetail}`)
          }
          break
        }
        default:
          break
      }
    })
    // 模型/工具层面的致命错误（不走 turn/end 的那种）
    offError = ctx.on('agent/error', (payload) => {
      if (payload?.agent !== a) return
      const detail = String(payload?.error?.message ?? payload?.error ?? 'unknown')
      lastTurnDetail = detail
      if (current) {
        current.progress.push({ at: Date.now(), text: `agent 错误: ${detail.slice(0, 300)}` })
      }
      log(`agent 错误: ${detail.slice(0, 300)}`)
    })
  }

  async function ensureAgent() {
    if (handle) return handle
    const sessionId = SessionId(`qqbot-relay-${randomUUID()}`)
    log(`创建受限 agent (tools=${config.allowedTools.join(',')}, cwd=${config.cwd})`)
    handle = await ctx.agents.create({
      sessionId,
      meta: { cwd: config.cwd },
      agentOptions: {
        provider: config.provider,
        model: config.model,
      },
      // setup 是唯一受支持的组合窗口：在这里把工具集收窄成白名单。
      setup: (agentCtx) => {
        agentCtx.tools.restrict({ allow: [...config.allowedTools] })
      },
    })
    agent = handle.agent
    attachProgress(agent)
    return handle
  }

  async function runTask(rec) {
    current = rec
    rec.state = 'running'
    rec.startedAt = Date.now()
    lastText = ''
    lastTurnKind = null
    lastTurnDetail = ''
    try {
      const h = await ensureAgent()
      const a = h.agent
      const fromSeq = a.session.seq
      a.followup(createUserMessage({
        content: [{ type: 'text', text: TASK_PROMPT(rec.task) }],
        source: { kind: 'plugin', plugin: name },
      }))
      const timer = setTimeout(() => {
        log(`任务 ${rec.id} 超时，取消`)
        try { a.cancel({ kind: 'user' }) } catch { /* ignore */ }
      }, config.taskTimeoutMs)
      try {
        await a.whenIdle()
      } finally {
        clearTimeout(timer)
      }
      const events = a.session.snapshotEvents(fromSeq)
      const raw = extractFinalText(events) || lastText
      const { headline, full } = splitResult(raw)
      rec.raw = raw
      rec.headline = headline
      rec.full = full

      // 先把 QQ 推送做完, **再**置终态 —— 否则语音侧一看到 done 就播报,
      // 会读到 qq=null(推 QQ 还在飞), 导致"详情已发你QQ"那句判断错。
      if (config.pushFullResult && openid && full) {
        try {
          rec.qq = await pushC2CMarkdown(openid, full, {
            appId: qqAppId, appSecret: qqAppSecret,
          })
          log(`任务 ${rec.id} 详情已推送 QQ(${rec.qq.chunks} 块)`)
        } catch (e) {
          rec.qqError = String(e?.message ?? e)
          log(`任务 ${rec.id} QQ 推送失败: ${rec.qqError}`)
        }
      }

      if (!raw && lastTurnKind && lastTurnKind !== 'completed') {
        // 没产出任何文本 + 本轮非正常结束 → 如实报错, 别假装成功
        if (lastTurnKind === 'aborted') {
          rec.state = 'cancelled'          // 用户主动取消, 不是故障
        } else {
          rec.state = 'error'
          rec.error = `turn ${lastTurnKind}: ${lastTurnDetail}`
        }
      } else {
        rec.state = 'done'
      }
      rec.endedAt = Date.now()
    } catch (err) {
      rec.state = 'error'
      rec.error = String(err?.message ?? err)
      rec.endedAt = Date.now()
      log(`任务 ${rec.id} 失败: ${rec.error}`)
    } finally {
      current = null
      const next = queue.shift()
      if (next) runTask(next)          // 单并发：上一个结束才跑下一个
    }
  }

  function submit(task) {
    const rec = {
      id: randomUUID().slice(0, 8), task, state: 'queued',
      progress: [], createdAt: Date.now(),
    }
    tasks.set(rec.id, rec)
    if (current) queue.push(rec)
    else runTask(rec)
    return rec
  }

  // ── 控制面：本地 HTTP ────────────────────────────────────────────────
  const readBody = (req) => new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (c) => {
      data += c
      if (data.length > 1e6) { reject(new Error('body too large')); req.destroy() }
    })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })

  const publicTask = (r) => ({
    id: r.id, task: r.task, state: r.state,
    headline: r.headline ?? '', full: r.full ?? '',
    progress: r.progress.slice(-20), error: r.error ?? null,
    qqError: r.qqError ?? null, qq: r.qq ?? null,
    startedAt: r.startedAt ?? null, endedAt: r.endedAt ?? null,
  })

  const server = http.createServer(async (req, res) => {
    const send = (code, obj) => {
      const body = JSON.stringify(obj, null, 2)
      res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' })
      res.end(body)
    }
    try {
      const url = new URL(req.url, 'http://localhost')
      if (req.method === 'GET' && url.pathname === '/health') {
        return send(200, {
          ok: true, agent: !!handle, running: current?.id ?? null,
          queued: queue.length, tasks: tasks.size,
          openidConfigured: !!openid,
          allowedTools: config.allowedTools,
        })
      }
      if (req.method === 'POST' && url.pathname === '/task') {
        const body = JSON.parse((await readBody(req)) || '{}')
        if (!body.task || !String(body.task).trim()) return send(400, { error: 'task 不能为空' })
        const rec = submit(String(body.task))
        log(`受理任务 ${rec.id}: ${String(body.task).slice(0, 60)}`)
        return send(202, { taskId: rec.id, state: rec.state })
      }
      if (req.method === 'GET' && url.pathname === '/tasks') {
        return send(200, { tasks: [...tasks.values()].map(publicTask) })
      }
      const m = url.pathname.match(/^\/task\/([\w-]+)$/)
      if (req.method === 'GET' && m) {
        const rec = tasks.get(m[1])
        return rec ? send(200, publicTask(rec)) : send(404, { error: 'not found' })
      }
      if (req.method === 'POST' && url.pathname === '/cancel') {
        if (current && agent) {
          agent.cancel({ kind: 'user' })
          return send(200, { cancelled: current.id })
        }
        return send(200, { cancelled: null })
      }
      return send(404, { error: 'unknown route' })
    } catch (e) {
      return send(500, { error: String(e?.message ?? e) })
    }
  })

  // 控制面：监听失败绝不能把整个 profile 拖down（这是用户 QQ 机器人的常驻进程）
  let listening = false
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject)
      server.listen(config.port, config.host, resolve)
    })
    listening = true
    log(`控制面就绪: http://${config.host}:${config.port}  (GET /health, POST /task, GET /task/:id, POST /cancel)`)
  } catch (e) {
    console.error(
      `[qqbot-relay] 控制面监听失败 (${config.host}:${config.port}): ${e?.message ?? e}` +
      ' —— 插件继续加载, QQ 机器人不受影响',
    )
  }

  // 插件卸载/进程退出时收尾
  ctx.effect(() => () => {
    try { offEvent?.() } catch { /* ignore */ }
    try { offError?.() } catch { /* ignore */ }
    if (listening) { try { server.close() } catch { /* ignore */ } }
    try { handle?.dispose() } catch { /* ignore */ }
    log('已停止')
  })
}
