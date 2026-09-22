# qqbot_relay —— 复杂路径委托中继（安装与验证手册）

挂在 **qqbot profile** 内（QQ 机器人进程本来就常驻，冷启动摊薄为零），把语音助手的复杂任务
交给一个**工具受限**的 dsh agent 处理，进度实时回流，完整报告主动推到 QQ。

## 组成

| 文件 | 作用 |
|---|---|
| `index.js` | Cordis 插件主体：本地 HTTP 控制面 + 驱动受限 agent + 消费会话事件做进度 + 完成时推 QQ |
| `qq_push.js` | QQ 官方「主动推送」客户端（零依赖，token 缓存 + 4500 字切块） |
| `cordis.patch.yml` | bundle patch：安装后自动挂载插件行（含默认白名单/端口） |
| `package.json` | 插件包声明（peerDependencies 与 `@tencent-connect/dsh-qqbot` 同款） |
| `peers.log` | **临时**：诊断补丁写入的入站 openid 记录（取到后回滚补丁并删除本文件） |

## 控制面 API（仅监听 127.0.0.1）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活、是否有 agent、是否有任务在跑、白名单工具 |
| POST | `/task` | body `{"task":"..."}` → `202 {"taskId":"..."}`（立即返回，异步执行） |
| GET | `/task/:id` | 任务状态、`headline`（口语结论）、`full`（完整报告）、`progress[]` |
| GET | `/tasks` | 全部任务 |
| POST | `/cancel` | 取消当前任务 |

## 安装（本机，profile = qqbot）

```powershell
# 1) 把插件链进 profile（dsh 会自动把它加进 dsh.profile.bundles）
dsh plugin --profile qqbot add "C:\Users\zj\Desktop\voice_assistant\bt_assistant\qqbot_relay"

# 2) 重启机器人（进程内加载插件）

# 3) 自检
curl http://127.0.0.1:8765/health
```

配置覆盖：在 `~/.dsh/profiles/qqbot/cordis.patch.yml` 里按 id `qqbot-relay` 覆盖，
需要填的是 `qqAppId` / `qqAppSecret` / `openid`（openid 留空则只在本地记账、不推送）。

## 工具白名单（安全边界）

`allowedTools` 默认 **只放行"查"的能力**：`web_search`、`web_fetch`、`read`、`glob`、`grep`。

不在白名单里的工具（`pwsh`/`bash`/`write`/`edit`/`str_replace_editor` 等）按 dsh 语义
**对该 agent 从 prompt 中消失且拒绝执行** —— 不是"提示模型别用"。语音误路由不会变成语音直通 shell。

## 双产物约定

一次模型输出里用 `===详情===` 分隔：前段是 1-3 句口语结论（喂 TTS），后段是完整报告（推 QQ）。
缺失分隔符时自动降级为"前两句当结论"，因此不会因模型不守约而丢结果。
