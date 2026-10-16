/**
 * 麦克风采集 worklet：把浏览器音频回调转成"16k / 16bit / 单声道 / 512 帧"的 PCM 块。
 *
 * 为什么用 AudioWorklet 而不是 ScriptProcessorNode：
 *   ScriptProcessor 跑在主线程，且已被标记废弃 —— 主线程一卡（渲染、GC）就丢音频，
 *   丢音频在本链路里等于"用户说的话缺一块"，会直接损坏 STT 与 VAD 判句。
 *   AudioWorklet 跑在专用音频线程，不受主线程影响。
 *
 * 为什么在这里做重采样/降混：手机麦克风通常是 48kHz（有时立体声），而整条链路统一
 * 16k/16bit/mono（与本机 `AudioDeviceAdapter` 一致，见 protocol/__init__.py）。
 * 在本线程做等于"零延迟成本"——数据还没过网络就已经是目标格式。
 *
 * 注意 **不做** 任何回声抑制：浏览器侧由 `getUserMedia({echoCancellation:true})`
 * 与"播放期间暂停采集"共同承担（见 app.js）。
 */

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512; // 32ms @16k

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || TARGET_RATE;
    this.frameSamples = opts.frameSamples || FRAME_SAMPLES;
    this.capturing = false;
    //: 回声抑制：true 时**继续接收音频但丢弃样本**。
    //: 为什么不干脆 stop：通话形态下会频繁暂停/恢复（助手每次说话），
    //: 反复启停会让音频线程与主线程之间产生状态抖动；只丢样本最稳。
    this.paused = false;
    this.buf = new Int16Array(this.frameSamples);
    this.filled = 0;
    // 重采样相位（保留小数部分，避免每块重新对齐造成的抖动）
    this.phase = 0;

    this.port.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === "start") {
        this.capturing = true;
        this.paused = false;
        this.filled = 0;
        this.phase = 0;
      } else if (msg.type === "stop") {
        this.capturing = false;
        this.flush(); // 把半块也发出去，否则句尾会丢
      } else if (msg.type === "pause") {
        this.paused = true;
        this.filled = 0;   // 丢弃半个未满的块，恢复后从干净边界开始
      } else if (msg.type === "resume") {
        this.paused = false;
      }
    };
  }

  flush() {
    if (this.filled > 0) {
      const out = this.buf.slice(0, this.filled);
      this.port.postMessage(out.buffer, [out.buffer]);
      this.filled = 0;
    }
  }

  process(inputs) {
    if (!this.capturing) return true;
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // 回声抑制期间：保持相位推进但**不产出样本**。
    // 相位也要推进，否则恢复采集时相位错位会造成一小段变速（听起来像"咔"一下）。
    if (this.paused) {
      const n = input[0].length;
      this.phase = (this.phase + n) % (sampleRate / this.targetRate || 1);
      return true;
    }

    const nCh = input.length;
    const n = input[0].length;
    const ctxRate = sampleRate; // AudioWorkletGlobalScope 提供的输入采样率
    const ratio = ctxRate / this.targetRate;

    // 逐输出样本：先对声道求平均，再按相位取输入样本（线性插值）
    let pos = this.phase;
    while (pos < n) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const i2 = Math.min(i + 1, n - 1);
      let s = 0;
      for (let c = 0; c < nCh; c++) {
        const a = input[c][i] || 0;
        const b = input[c][i2] || 0;
        s += a * (1 - frac) + b * frac;
      }
      s /= nCh;

      // 裁剪 → int16
      let v = s < -1 ? -1 : s > 1 ? 1 : s;
      this.buf[this.filled++] = v < 0 ? v * 0x8000 : v * 0x7fff;

      if (this.filled === this.frameSamples) {
        const out = this.buf.slice(0); // 复制：buffer 要转移所有权给主线程
        this.port.postMessage(out.buffer, [out.buffer]);
        this.filled = 0;
      }
      pos += ratio;
    }
    this.phase = pos - n; // 跨块保留相位

    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
