"""自签 TLS 证书：让局域网里的手机能用 https/wss 访问。

## 为什么必须要 TLS（不只是"更安全"）

浏览器只在**安全上下文**里把麦克风给页面。``http://192.168.0.8:8801`` 属于不安全
来源（``http://localhost`` 是例外），Chrome 会**直接拒绝 getUserMedia** —— 现象是
"按住说话没反应"，很容易被误判成代码坏了（实测踩到）。所以手机端要能用，TLS 不是
可选项。

顺带解决第二件事：现在局域网里的音频是**明文 PCM**，TLS 之后才不外泄。

## 为什么用自签证书而不是 Let's Encrypt

局域网 IP（192.168.x.x）签不出公信证书，内网用 IP 访问也没法用域名校验。
所以：自签 + 手机首次点一次"继续访问"。这是内网方案的固有代价。

证书里**必须带 IP SAN**：只写 CN 的话，现代浏览器（Chrome 58+ 起）会因为
"证书缺少使用者备用名称"而**拒绝连接**（不是只警告），这是常见坑。

## 依赖

需要 ``cryptography``（比调 openssl 命令行更可靠：不依赖外部程序、跨平台一致）。
"""

import datetime
import ipaddress
import logging
import os
import socket
import ssl
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "certs"
DEFAULT_HOSTS = ("localhost", "127.0.0.1", "::1")


def detect_local_ip() -> str:
    """本机在局域网里的 IPv4（用 UDP connect 探测默认出口，不真发包）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 53))
        return s.getsockname()[0]
    except Exception:                                             # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def hostnames_for_cert(extra: Iterable[str] = ()) -> List[str]:
    """证书要覆盖的主机名/IP 列表：默认包含常见本机地址 + 自动探测的局域网 IP。"""
    names = list(DEFAULT_HOSTS)
    for cand in (detect_local_ip(), socket.gethostname()):
        if cand and cand not in names:
            names.append(cand)
    for e in extra:
        if e and e not in names:
            names.append(e)
    return names


def _generate_cert(cert_path: Path, key_path: Path, hosts: List[str]) -> None:
    """生成自签证书（RSA 2048 / SHA256 / 825 天）。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # 主机名进 DNS SAN，IP 进 IP SAN —— 缺了 IP SAN，Chrome 会直接拒绝
    dns, ips = [], []
    for h in hosts:
        try:
            ips.append(ipaddress.ip_address(h))
        except ValueError:
            dns.append(h)

    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, hosts[0] if hosts else "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "voice-assistant-local"),
    ])
    # 主机名进 DNS SAN，IP 进 IP SAN —— 缺了 IP SAN，Chrome 会**直接拒绝**连接
    san_entries = [x509.DNSName(d) for d in dns] + [x509.IPAddress(i) for i in ips]

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)                    # 自签：签发者 = 主体
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    # ⚠️ 只在 POSIX 上收敛私钥权限。Windows 的 chmod 只映射"只读"位，
    # 对目录/文件设 0o600 会把它变成**拒绝访问**（实测把父目录也一起弄坏，
    # 连临时目录都删不掉）。Windows 上依赖 ACL 即可，不做画蛇添足的 chmod。
    if os.name == "posix":
        try:
            key_path.chmod(0o600)
        except OSError:
            pass


def ensure_self_signed(
    directory: Optional[Path] = None,
    hosts: Iterable[str] = (),
    force: bool = False,
    debug: bool = False,
) -> Tuple[Path, Path]:
    """确保存在可用的自签证书，返回 ``(cert_path, key_path)``。

    已存在且未过期就直接复用（否则手机每次都要重新信任一次，很烦）。
    """
    d = Path(directory) if directory else DEFAULT_DIR
    cert_path, key_path = d / "server.crt", d / "server.key"

    if not force and cert_path.is_file() and key_path.is_file():
        if _not_expired(cert_path):
            if debug:
                logger.info("复用自签证书: %s", cert_path)
            return cert_path, key_path
        logger.info("自签证书已过期，重新生成")

    want = hostnames_for_cert(hosts)
    try:
        _generate_cert(cert_path, key_path, want)
    except ImportError as ex:
        raise RuntimeError(
            "生成自签证书需要 cryptography：pip install cryptography "
            f"（或自备证书到 {cert_path} / {key_path}）。原始错误: {ex}"
        ) from ex
    logger.info("已生成自签证书: %s（覆盖 %s）", cert_path, ", ".join(want))
    return cert_path, key_path


def _not_expired(cert_path: Path) -> bool:
    """用标准库检查过期（避免为了这一步强依赖 cryptography）。"""
    try:
        import ssl as _ssl
        info = _ssl._ssl._test_decode_cert(str(cert_path))     # type: ignore[attr-defined]
        not_after = info.get("notAfter")
        if not not_after:
            return True
        exp = datetime.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
        return exp > datetime.datetime.utcnow() + datetime.timedelta(days=1)
    except Exception:                                             # noqa: BLE001
        return True                       # 判不了就当作可用（真坏了会在握手时报错）


def build_ssl_context(cert_path: Path, key_path: Path) -> "ssl.SSLContext":
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def print_fingerprint(cert_path: Path) -> str:
    """打印证书指纹：手机上要核对"你连的是不是这台机器"时用。"""
    try:
        data = cert_path.read_bytes()
        import hashlib
        der = ssl.PEM_cert_to_DER_cert(data.decode("ascii"))
        fp = hashlib.sha256(der).hexdigest()
        return ":".join(fp[i:i + 2] for i in range(0, 16, 2)).upper()
    except Exception:                                             # noqa: BLE001
        return "?"
