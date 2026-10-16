#!/usr/bin/env python3
"""只读诊断：你的 NAS 公网入口是哪一种？

为什么需要它：手机接语音助手需要一条"公网可达"的通道，而通道形态
（IPv4 端口映射 / IPv6 / 反代 / 隧道 / 厂商中转 / 组网 VPN）决定了服务端
要不要自己做 TLS、能不能随便加 WebSocket 长连。这些信息只能从你本机实测。

用法：
    python tools/probe_ingress.py <域名|IP|完整URL> [端口...]

例：
    python tools/probe_ingress.py https://yyssccy.fnos.net/
    python tools/probe_ingress.py yyssccy.fnos.net 443

**只做连接测试，不登录、不发送任何数据、不修改任何配置。**
"""

import os
import socket
import struct
import sys
import urllib.parse

#: 用公共 DNS 复测一次，排除"本机 DNS 走了代理/DNS 被劫持"造成的假否定
PUBLIC_DNS = "223.5.5.5"


def normalize_host(raw: str) -> str:
    """把用户可能粘进来的完整 URL / 带端口 / 带路径的写法收敛成纯主机名。

    第一版忘了这一步：用户粘 ``https://yyssccy.fnos.net/`` 时，脚本把整个
    字符串丢给 getaddrinfo，得到 ``getaddrinfo failed`` —— 那是"这不是合法
    主机名"，**不是"域名不存在"**，结论会完全反过来。
    """
    s = (raw or "").strip().strip('"').strip("'")
    if "://" in s:
        s = urllib.parse.urlsplit(s).netloc or s.split("://", 1)[1]
    s = s.split("/")[0].split("?")[0].split("#")[0]
    if s.count(":") == 1 and not s.endswith("]"):        # host:port
        s = s.split(":")[0]
    s = s.strip().strip("[]")
    # 去端口后仍失败时，保留原始输入供报错展示
    return s

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 本机局域网出口（要让 NAS 反代能回连的那台 PC）
LOCAL_PC = "192.168.0.8"
LOCAL_PORT = 8765  # 中继控制面（当前仅监听回环，此处只作为连通性参考）


def resolve(name: str):
    """解析域名的 A / AAAA 记录 —— IPv6 还是 IPv4 决定通道形态。"""
    print(f"\n=== DNS 解析: {name} ===")
    found = {"v4": [], "v6": []}
    try:
        for info in socket.getaddrinfo(name, None):
            fam = info[0]
            addr = info[4][0]
            key = "v4" if fam == socket.AF_INET else "v6"
            if addr not in found[key]:
                found[key].append(addr)
    except Exception as ex:
        print(f"  解析失败: {type(ex).__name__}: {ex}")
        return found
    if found["v4"]:
        print("  IPv4:", ", ".join(found["v4"]))
    else:
        print("  IPv4: 无记录")
    if found["v6"]:
        print("  IPv6:", ", ".join(found["v6"]), "  ← 有 AAAA，说明可能是 IPv6 直连方案")
    else:
        print("  IPv6: 无记录")
    return found


def dns_query(name: str, qtype: int, server: str = PUBLIC_DNS, timeout: float = 4.0):
    """用指定 DNS 服务器直接查 A / AAAA / CNAME（不依赖本机 DNS 配置）。

    为什么要绕过本机解析器：本机流量走代理（实测 :7892），系统 DNS 的解析结果
    未必代表手机在外网看到的解析结果 —— 而"手机看到什么"才是我们要判断的。
    """
    tid = 0x1234
    q = struct.pack(">HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    for part in name.split("."):
        q += bytes([len(part)]) + part.encode()
    q += b"\x00" + struct.pack(">HH", qtype, 1)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(q, (server, 53))
        data, _ = s.recvfrom(4096)
    except Exception as ex:                                       # noqa: BLE001
        return f"查询失败: {type(ex).__name__}: {ex}"
    finally:
        s.close()

    if len(data) < 12:
        return "响应过短"
    rcode = data[3] & 0x0F
    if rcode == 3:
        return "NXDOMAIN（该名字不存在）"
    if rcode != 0:
        return f"DNS 错误码 rcode={rcode}"

    ancount = struct.unpack(">H", data[6:8])[0]
    # 跳过 question 段
    off = 12
    while off < len(data) and data[off]:
        off += data[off] + 1
    off += 5

    out = []
    for _ in range(ancount):
        if off + 12 > len(data):
            break
        # 名字可能是压缩指针
        while off < len(data) and data[off] & 0xC0 == 0xC0:
            off += 2
            break
        else:
            while off < len(data) and data[off]:
                off += data[off] + 1
            off += 1
        if off + 10 > len(data):
            break
        rtype, _, _, rdlen = struct.unpack(">HHIH", data[off:off + 10])
        off += 10
        rdata = data[off:off + rdlen]
        off += rdlen
        if rtype == 1 and rdlen == 4:
            out.append("A " + ".".join(str(b) for b in rdata))
        elif rtype == 28 and rdlen == 16:
            out.append("AAAA " + ":".join(f"{rdata[i]:02x}{rdata[i+1]:02x}" for i in range(0, 16, 2)))
        elif rtype == 5:
            # CNAME：解出真实目标（这里只做粗糙解码，够用于判断"是否中转"）
            try:
                lbls, i = [], 0
                while i < len(rdata) and rdata[i]:
                    n = rdata[i]
                    lbls.append(rdata[i + 1:i + 1 + n].decode(errors="replace"))
                    i += n + 1
                out.append("CNAME " + ".".join(lbls))
            except Exception:                                     # noqa: BLE001
                out.append("CNAME (解析跳过)")
    return ", ".join(out) if out else "无记录"


def probe(host: str, port: int, timeout: float = 6.0) -> str:
    """尝试建立一个 TCP 连接然后立刻关闭（不发送任何应用层数据）。"""
    label = f"{host}:{port}"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "开放/可达"
    except socket.timeout:
        return f"超时（可能被防火墙丢弃）: {label}"
    except ConnectionRefusedError:
        return "连接被拒绝（主机在，但该端口没服务）"
    except OSError as ex:
        return f"失败: {type(ex).__name__}: {ex}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    host = normalize_host(sys.argv[1])
    ports = [int(p) for p in sys.argv[2:]] or [443]

    print("=" * 70)
    print("NAS 公网入口诊断（只读）")
    print("=" * 70)
    print(f"原始输入: {sys.argv[1]}")
    print(f"归一化主机名: {host}")

    resolve(host)

    print(f"\n=== 公共 DNS 直查（绕过本机解析器与代理）: {host} ===")
    for label, qtype in (("A", 1), ("AAAA", 28), ("CNAME", 5)):
        print(f"  {label:<6} {dns_query(host, qtype)}")

    print(f"\n=== 端口连通性: {host} ===")
    for p in ports:
        print(f"  {p:>5}  {probe(host, p)}")

    # 本机出口（对比用：确认本机能不能出公网，以及是否走了代理）
    print("\n=== 本机出口对照 ===")
    for target, port in (("223.5.5.5", 53), ("1.1.1.1", 443)):
        print(f"  直连 {target}:{port}  {probe(target, port, timeout=4)}")
    print("  注：若直连超时但浏览器正常，说明本机流量走代理（实测代理在 :7892）")

    print("\n=== 本机局域网 ===")
    print(f"  本机 LAN IP: {LOCAL_PC}（NAS 反代的目标地址）")

    print("""
----------------------------------------------------------------------
请把以上完整输出贴回给我。重点看三处：

1. DNS 里有没有 IPv6（AAAA）→ 决定是「IPv6 直连」还是「IPv4 端口映射/反代」
2. 443 是否可达 → 决定 NAS 能不能直接做 HTTPS 入口
3. 实际开放的是哪个端口 → 反代规则就照它加

如果你用的是 VPN / 组网工具（Tailscale、ZeroTier、WireGuard、frp、
Cloudflare Tunnel、飞牛的远程访问功能等），请直接告诉我工具名 ——
那比对端口更有用，因为有些通道（尤其厂商中转）不能随便加 WebSocket 长连。
----------------------------------------------------------------------""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
