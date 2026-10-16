#!/usr/bin/env python3
"""临时探针服务：用来判断"公网入口会不会把请求转发到这台 PC"。

原理（不碰任何既有配置）：
    在 PC 上监听 8899，任何请求都返回一段**带上"我是谁"标记**的 JSON。
    然后从公网访问 NAS 的入口（`yyssccy.fnos.net`）的各种候选路径：
      · 若响应里出现本探针的标记 → **公网入口确实转发到了这台 PC**
        → 说明飞牛网关支持"反代到内网任意设备"，方案①成立
      · 若响应是飞牛自己的页面/404 → 说明只服务 NAS 自身

为什么用 JSON 标记而不是靠状态码：网页界面到处都返回 200，
只有"响应体里出现本机独有标记"才是可靠证据。

用法::

    python tools/probe_relay_target.py --port 8899        # 起探针（前台，Ctrl+C 停）
    python tools/probe_relay_target.py --check            # 另开一个窗口：探测公网路径
"""

import argparse
import json
import os
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: 本机独有标记 —— 公网入口转发过来时，这个字符串必须原样出现在响应体里
MARKER = "VOICE-ASSISTANT-PC-PROBE-192-168-0-8"

#: 公网入口（用户 NAS 的 fnos.net 域名）
PUBLIC_HOST = "yyssccy.fnos.net"

#: 线上探测的候选路径：网关若按**路径前缀**转发，这些是最可能被占用的
CANDIDATE_PATHS = [
    "/", "/probe", "/voice", "/voice/", "/app", "/apps", "/proxy", "/relay",
    "/api", "/api/probe", "/ws", "/chat", "/assistant", "/xiaoniu",
    "/8899", "/port/8899", "/p/8899",
]

history = []
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self):
        with lock:
            history.append({
                "at": time.strftime("%H:%M:%S"),
                "method": self.command,
                "path": self.path,
                "host": self.headers.get("Host", ""),
                "xff": self.headers.get("X-Forwarded-For", ""),
                "ua": (self.headers.get("User-Agent") or "")[:60],
                "peer": self.client_address[0],
            })
        body = json.dumps({
            "marker": MARKER,
            "who": "PC 192.168.0.8 上的探针服务",
            "path": self.path,
            "host_header": self.headers.get("Host", ""),
            "x_forwarded_for": self.headers.get("X-Forwarded-For", ""),
            "peer": self.client_address[0],
            "note": "看到这段 JSON 就说明：公网入口把请求转发到了这台 PC",
        }, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Probe-Marker", MARKER)
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_HEAD = _reply

    def log_message(self, fmt, *args):        # 静默默认日志（我们自己打印）
        pass


def serve(port: int):
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"✅ 探针已启动：http://0.0.0.0:{port}  （标记 {MARKER}）")
    print("   本机自测：", end="", flush=True)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/self", timeout=3) as r:
            print("OK" if MARKER in r.read().decode() else "异常")
    except Exception as ex:                                        # noqa: BLE001
        print(f"失败 {ex}")

    def reporter():
        last = 0
        while True:
            time.sleep(2)
            with lock:
                new = history[last:]
                last = len(history)
            for h in new:
                print(f"  📥 [{h['at']}] {h['method']} {h['path']}  "
                      f"Host={h['host']}  XFF={h['xff'] or '-'}  peer={h['peer']}")

    threading.Thread(target=reporter, daemon=True).start()
    print("\n等待外部请求…（Ctrl+C 停止）")
    print("→ 现在在**另一个窗口**执行：python tools/probe_relay_target.py --check\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        srv.server_close()


def _fetch(url: str, timeout: float = 8.0):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "probe-relay/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            body = r.read(20000).decode("utf-8", "replace")
            return r.status, body, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(4000).decode("utf-8", "replace"), dict(e.headers)
    except Exception as ex:                                        # noqa: BLE001
        return None, f"{type(ex).__name__}: {ex}", {}


def check(host: str):
    print("=" * 72)
    print(f"从公网探测 {host}：请求会不会被转发到这台 PC（标记 {MARKER}）")
    print("=" * 72)

    # 先确认域名解析到哪
    try:
        ip = socket.gethostbyname(host)
        print(f"\nDNS: {host} → {ip}")
    except Exception as ex:                                        # noqa: BLE001
        print(f"\nDNS 解析失败：{ex}")
        return 1

    print("\n【A】根路径（应返回飞牛页面，用于对照）")
    st, body, hdr = _fetch(f"https://{host}/")
    hit = MARKER in body
    print(f"  / → HTTP {st}  Server={hdr.get('Server')}  长度={len(body)}  "
          f"{'🎯 命中本机探针！' if hit else '（飞牛自己的页面）'}")

    print("\n【B】候选路径（若网关按路径转发，这里会出现命中）")
    hits = []
    for p in CANDIDATE_PATHS:
        st, body, hdr = _fetch(f"https://{host}{p}", timeout=6)
        hit = MARKER in body
        kind = "🎯 命中探针" if hit else ("飞牛页面" if "飞牛" in body[:600] else f"其他({len(body)}B)")
        print(f"  {p:<14} → HTTP {st:<5} {kind}")
        if hit:
            hits.append(p)

    print("\n" + "=" * 72)
    if hits:
        print(f"✅ 结论：公网入口**会**把请求转发到这台 PC —— 命中路径 {hits}")
        print("   → 飞牛网关支持反代到内网其他设备：语音服务可以留在 PC，NAS 只当入口")
        return 0
    print("❌ 结论：候选路径**没有**转发到这台 PC")
    print("""
   但这还不是最终结论，有两种可能：
     ① 网关不支持转发到内网其他设备（只服务 NAS 自身）
     ② 支持，但需要你在网关里**显式添加**一条规则（路径名不在这批候选里）

   如何区分：去 fnOS「设置 → 远程访问 / 统一网关」里看有没有"添加目标/反代"入口。
        · 有 → 你加一条指向 192.168.0.8:8899 的规则，然后重跑本检查（用你起的路径名）
        · 没有 → 就是 ①，语音服务必须搬到 NAS 上""")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--check", action="store_true", help="从公网探测（另开窗口执行）")
    ap.add_argument("--host", default=PUBLIC_HOST)
    args = ap.parse_args()
    if args.check:
        return check(args.host)
    serve(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
