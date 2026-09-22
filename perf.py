"""性能基线采集：把每轮的时延按**路径**归类，累积落盘 + 结束时汇总。

为什么要分路径：快路径（纯知识问答）和复杂路径（委托）的时延语义完全不同，
混在一起统计会让基线失去意义 —— 复杂路径看的应该是"说话结束→**确认话术**出声"，
而不是"→最终答案出声"（那可能是几十秒后）。

口径（pipeline 打点）：
  vad_to_stt_ms            VAD 判句 → STT 完成
  stt_to_llm_first_ms      STT 完成 → LLM 首 token（TTFT）
  llm_first_to_tts_first_ms  首 token → 首个音频包
  vad_to_tts_first_ms      **说话结束 → 出声**（用户感知总时延）
  path                     chat / tool / delegate
"""

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PerfRecorder:
    """按路径累积时延样本，落 JSONL 并在结束时打印汇总。"""

    PATH_LABEL = {
        "chat": "快路径/纯知识",
        "tool": "快路径/本地工具",
        "delegate": "复杂路径/委托",
    }

    def __init__(self, jsonl_path: Optional[str] = None, debug: bool = False):
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.debug = debug
        self.samples: List[Dict] = []
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, request, metrics: Dict):
        """pipeline 的 on_performance 回调。"""
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "path": metrics.get("path", "chat"),
            "text": (metrics.get("text") or "")[:40],
            "tools": metrics.get("tools") or [],
            "vad_to_stt_ms": metrics.get("vad_to_stt_ms"),
            "stt_to_llm_request_ms": metrics.get("stt_to_llm_request_ms"),
            "llm_request_to_token_ms": metrics.get("llm_request_to_token_ms"),
            "stt_to_llm_token_ms": metrics.get("stt_to_llm_token_ms"),
            "llm_token_to_sentence_ms": metrics.get("llm_token_to_sentence_ms"),
            "stt_to_llm_first_ms": metrics.get("stt_to_llm_first_ms"),
            "llm_first_to_tts_first_ms": metrics.get("llm_first_to_tts_first_ms"),
            "vad_to_tts_first_ms": metrics.get("vad_to_tts_first_ms"),
        }
        self.samples.append(row)
        if self.debug:
            logger.info(
                f"[基线] {row['path']:8s} 说话结束→出声 {row['vad_to_tts_first_ms']}ms "
                f"(STT {row['vad_to_stt_ms']} + 请求前 {row['stt_to_llm_request_ms']} "
                f"+ 请求→token {row['llm_request_to_token_ms']} "
                f"+ 等到句末 {row['llm_token_to_sentence_ms']} "
                f"+ TTS首包 {row['llm_first_to_tts_first_ms']}) \"{row['text']}\""
            )
        if self.jsonl_path:
            try:
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as ex:
                logger.warning(f"写基线文件失败: {ex}")

    def summary(self) -> str:
        """按路径汇总（中位数/最小/最大/样本数）。"""
        if not self.samples:
            return "（本次没有样本）"
        lines = ["", "=" * 78, "时延基线汇总（单位 ms，按路径分组）", "=" * 78]
        head = f"{'路径':12s}{'样本':>5s}{'总时延中位':>10s}{'最短':>7s}{'最长':>7s}" \
               f"{'STT':>6s}{'请求前':>7s}{'请求→token':>11s}{'token→句':>9s}{'LLM首句':>8s}{'TTS首包':>8s}"
        lines.append(head)
        lines.append("-" * len(head))
        for path in ("chat", "tool", "delegate"):
            rows = [r for r in self.samples if r["path"] == path]
            if not rows:
                continue

            def med(key):
                vals = [r[key] for r in rows if r.get(key) is not None]
                return round(statistics.median(vals)) if vals else None

            totals = [r["vad_to_tts_first_ms"] for r in rows
                      if r.get("vad_to_tts_first_ms") is not None]
            lines.append(
                f"{self.PATH_LABEL[path]:12s}{len(rows):>5d}"
                f"{(round(statistics.median(totals)) if totals else 'n/a'):>10}"
                f"{(min(totals) if totals else 'n/a'):>7}"
                f"{(max(totals) if totals else 'n/a'):>7}"
                f"{str(med('vad_to_stt_ms')):>6}"
                f"{str(med('stt_to_llm_request_ms')):>7}"
                f"{str(med('llm_request_to_token_ms')):>11}"
                f"{str(med('llm_token_to_sentence_ms')):>9}"
                f"{str(med('stt_to_llm_first_ms')):>8}"
                f"{str(med('llm_first_to_tts_first_ms')):>8}"
            )
        lines.append("-" * len(head))
        lines.append("注：复杂路径的\"总时延\"是**确认话术出声**；最终答案属异步播报，不计入。")
        lines.append("    STT=语音识别；请求前=停播放/组prompt等本地开销；请求→token=网络+模型开口；")
        lines.append("    token→句=等到句末标点才能合成（实测仅 ~100ms）。")
        return "\n".join(lines)

    def print_summary(self):
        text = self.summary()
        print(text)
        logger.info(text)
