#!/usr/bin/env python3
"""外网可达性诊断：判断"手机从外网能不能连到本机的服务端口"。

**只做只读探测**：不修改任何配置、不登录任何设备、不发送应用层数据。
只回答四个问题：

1. 本机的公网出口 IP 是什么（对比本机 LAN 地址，判断是否在 NAT/CGNAT 后）
2. 从公网能不能连到家里几个候选端口（80 / 443 / 8443 / 9443 / 8801）
3. 有没有 IPv6 出网（IPv6 是绕开端口映射的另一条路）
4. 本机有没有 UPnP（有些路由器允许程序自己开映射）

用法::

    python tools/probe_external.py
    python tools/probe_external.py --ports 80 443 8443 8801

判读要点见输出末尾的说明。
"""

import argparse
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

#: 取公网 IP 的服务（挑国内可达的；失败会自动换下一个）
IP_SERVICES = [
    "https://api.ipify.org?format=json",
    "https://ifconfig.me/ip",
    "https://ipinfo.io/json",
]

#: 本机服务端口（测"外网能不能连回家"时用它）
LOCAL_PORT = 8801


def _e(url: str, timeout: float = 8.0):
    """绕开本机代理取内容（本机开着代理会让直连结果失真）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace").strip()


def public_ipv4():
    for svc in IP_SERVICES:
        try:
            raw = _e(svc)
            if "{" in raw:
                data = json.loads(raw)
                for k in ("ip", "query"):
                    if data.get(k):
                        return data[k]
                return None
            return raw.split()[0]
        except Exception:                                          # noqa: BLE001
            continue
    return None


def lan_ipv4() -> str:
    """本机在局域网里的地址（UDP connect 探测默认出口，不真发包）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 53))
        return s.getsockname()[0]
    except Exception:                                              # noqa: BLE001
        return "?"


def tcp_check(host: str, port: int, timeout: float = 6.0) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "可达"
    except socket.timeout:
        return "超时（被丢弃/未映射）"
    except ConnectionRefusedError:
        return "拒绝（到达了主机，但端口没服务）"
    except OSError as ex:
        return f"失败 {type(ex).__name__}"


def tls_probe(host: str, port: int, timeout: float = 8.0):
    """取对端证书信息 —— 用来判断那个端口后面是谁（NAS？路由器？）。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            with ctx.wrap_socket(s, server_hostname=host) as ss:
                cert = ss.getpeercert()
                if not cert:
                    return "TLS 握手成功但证书为空（可能不是标准 HTTPS 服务）"
                sub = dict(x[0] for x in cert.get("subject", ()))
                iss = dict(x[0] for x in cert.get("issuer", ()))
                return (f"CN={sub.get('commonName')} | 签发者={iss.get('organizationName') or iss.get('commonName')}"
                        f" | 有效期至 {cert.get('notAfter')}")
    except Exception as ex:                                        # noqa: BLE001
        return f"TLS 探测失败：{type(ex).__name__}: {str(ex)[:80]}"


def ipv6_ok(timeout: float = 6.0) -> str:
    """有没有 IPv6 出网（有的话可以完全绕开端口映射）。"""
    try:
        infos = socket.getaddrinfo("ipv6.google.com", 443, socket.AF_INET6)
        if not infos:
            return "本机无 IPv6 地址"
        addr = infos[0][4][0]
    except Exception:                                              # noqa: BLE001
        return "DNS 无 AAAA 记录（本机可能没有 IPv6）"
    try:
        with socket.create_connection((addr, 443), timeout=timeout):
            return f"✅ IPv6 可出网（{addr}）→ 可考虑 IPv6 直连方案"
    except Exception as ex:                                        # noqa: BLE001
        return f"有 IPv6 地址但连不出去（{type(ex).__name__}）"


def upnp_available(timeout: float = 3.0) -> str:
    """SSDP 探测有没有 UPnP 设备（有些路由器允许程序自建端口映射）。"""
    msg = ("M-SEARCH * HTTP/1.1\r\n"
           "HOST: 239.255.255.250:1900\r\n"
           'MAN: "ssdp:discover"\r\n'
           "MX: 2\r\n"
           "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n").encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(msg, ("239.255.255.250", 1900))
        data, addr = s.recvfrom(65507)
        return f"发现 UPnP 设备（来自 {addr[0]}）"
    except socket.timeout:
        return "未发现 UPnP（或路由器关闭了 SSDP）"
    except OSError as ex:
        return f"UPnP 探测失败：{ex}"
    finally:
        s.close()


def local_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except Exception:                                              # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", type=int, nargs="*", default=[80, 443, 8443, 9443, LOCAL_PORT],
                    help="要检测的公网端口（默认 80 443 8443 9443 8801）")
    ap.add_argument("--timeout", type=float, default=6.0)
    args = ap.parse_args()

    print("=" * 70)
    print("外网可达性诊断（只读；不修改任何配置）")
    print("=" * 70)

    lan = lan_ipv4()
    pub = public_ipv4()
    print(f"\n【1】地址")
    print(f"   本机局域网地址 : {lan}")
    print(f"   公网出口 IP    : {pub or '取不到（网络受限）'}")
    print(f"   本机服务端口   : {LOCAL_PORT} → {'在监听' if local_listening(LOCAL_PORT) else '未监听（先起服务再测）'}")

    if pub:
        print(f"\n【2】从公网回连（目标 {pub}）")
        print("    注：部分家用路由器不支持 NAT 回环（hairpin），本机自己探测可能失败，")
        print("        这**不代表**外网连不上 —— 最终判据是手机切 4G 实测。")
        for p in args.ports:
            res = tcp_check(pub, p, args.timeout)
            line = f"    {p:>5} : {res}"
            if res == "可达" and p in (443, 8443, 9443):
                line += f"\n            └─ 该端口后面是谁：{tls_probe(pub, p, args.timeout)}"
            print(line)

    print(f"\n【3】IPv6")
    print(f"   {ipv6_ok()}")

    print(f"\n【4】UPnP")
    print(f"   {upnp_available()}")

    print("""
======================================================================
怎么读这份结果：

  · 公网 IP 与本机 LAN 地址不同  → 正常，说明有 NAT（家用宽带必然如此）
  · 端口列表中**有任何一个"可达"** → 说明运营商**放行了入站**、路由器也在转发，
      那么"再加一条映射到本机 8801"这条路大概率可行（去路由器确认即可）
  · 端口全部"超时" → 两种可能：① 没有公网 IP（CGNAT）② 有公网 IP 但没做映射
      区分方法：登录路由器看 WAN 口 IP；若它 != 上面的公网出口 IP，就是 CGNAT
      → 此时端口映射这条路走不通，改用隧道（Cloudflare Tunnel / frp / cpolar）或 IPv6
  · IPv6 可出网 → 另一条完全不用端口映射的路（但手机侧也要有 IPv6）

最终判据只有一条：**手机切到 4G/5G（关 WiFi）能不能打开页面**。
本机探测受 NAT 回环影响，只能作为参考。
======================================================================""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
