"""API 并发测试：起真实服务进程，并发打 /v1/completions。

验证三件事：结果正确不串话、多次并发结果稳定、并发确实比串行快。

用法: cd tests && python api_test.py
"""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

MODEL = "/mnt/mizar/models/Qwen3-0.6B-Base"
PROMPTS = [
    "what is the capital of France?",
    "what is the largest animal in the world?",
    "who wrote Romeo and Juliet?",
]
# 内容断言关键词：不串话 = 每条回答包含自己问题的答案。
KEYWORDS = {
    "what is the capital of France?": "Paris",
    "what is the largest animal in the world?": "blue whale",
    "who wrote Romeo and Juliet?": "Shakespeare",
}
MAX_TOKENS = 48


def free_port() -> int:
    """让系统分配一个空闲端口。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def completion(port: int, prompt: str, temperature: float) -> tuple[str, float]:
    """打一次 /v1/completions。

    Args:
        port: 服务端口。
        prompt: 输入 prompt。
        temperature: 采样温度，0 表示 greedy。

    Returns:
        (生成文本, 耗时秒数)。
    """
    body = {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = json.loads(resp.read())["choices"][0]["text"]
    return text, time.perf_counter() - start


def run_concurrent(port: int) -> tuple[dict, float]:
    """三路并发跑一遍 prompt，返回结果和墙钟耗时。"""
    results = {}

    def worker(prompt: str) -> None:
        results[prompt] = completion(port, prompt, 0)[0]

    start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(p,)) for p in PROMPTS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, time.perf_counter() - start


def main() -> None:
    """跑完全部并发检查。"""
    port = free_port()
    log_file = tempfile.NamedTemporaryFile(mode="w+",
                                           suffix=".log",
                                           delete=False)
    server = subprocess.Popen(
        [
            sys.executable, "-m", "lazy.api_server", "--model", MODEL, "--host",
            "127.0.0.1", "--port",
            str(port)
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={
            **os.environ, "LAZY_LOG_LEVEL": "DEBUG"
        },
    )
    try:
        # 等服务就绪（模型加载约 10s）。
        deadline = time.time() + 180
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                       timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not become ready in 180s")

        # 1. 单跑基线（greedy，确定），顺带预热。
        serial_sum = 0.0
        for prompt in PROMPTS:
            text, elapsed = completion(port, prompt, 0)
            serial_sum += elapsed
            assert text.strip(), f"empty output for {prompt!r}"
            keyword = KEYWORDS[prompt]
            assert keyword in text, (
                f"single run missing {keyword!r}: {text!r}")

        # 2. 三路并发跑两轮：内容正确（不串话）+ 跨轮逐字一致。
        # 不能断言"并发 == 单跑"：batched matmul 归约顺序不同，bf16 近同
        # 分点翻牌是预期行为，正确性按关键词和跨轮稳定性把关。
        round1, wall = run_concurrent(port)
        round2, _ = run_concurrent(port)
        for prompt in PROMPTS:
            keyword = KEYWORDS[prompt]
            assert keyword in round1[prompt], (
                f"concurrent wrong content for {prompt!r}: {round1[prompt]!r}")
            assert round1[prompt] == round2[prompt], (
                f"concurrent result not deterministic for {prompt!r}\n"
                f"round1: {round1[prompt]!r}\nround2: {round2[prompt]!r}")
        assert wall < 0.8 * serial_sum, (
            f"no concurrency: wall {wall:.1f}s vs serial {serial_sum:.1f}s")

        # 3. 混合温度并发：greedy 走 argmax 不崩（除 0 保护），采样结果非空。
        mixed = {}

        def worker_mixed(prompt: str, temp: float, key: str) -> None:
            mixed[key] = completion(port, prompt, temp)[0]

        greedy_thread = threading.Thread(target=worker_mixed,
                                         args=(PROMPTS[0], 0, "greedy"))
        sampled_thread = threading.Thread(target=worker_mixed,
                                          args=(PROMPTS[1], 1.0, "sampled"))
        greedy_thread.start()
        sampled_thread.start()
        greedy_thread.join()
        sampled_thread.join()
        assert KEYWORDS[PROMPTS[0]] in mixed["greedy"], (
            f"mixed greedy wrong: {mixed['greedy']!r}")
        assert mixed["sampled"].strip(), "sampled request returned empty"

        # 4. 服务日志应出现拼 batch 的 decode。
        server.terminate()
        server.wait(timeout=30)
        log_file.seek(0)
        batched = re.findall(r"decode: (\d+) seqs batched", log_file.read())
        assert any(int(n) >= 2 for n in batched), (
            f"no batched decode in server log: {batched}")

        print("all api checks passed")
        print(f"  serial sum: {serial_sum:.1f}s, concurrent wall: {wall:.1f}s, "
              f"max batched: {max(map(int, batched))}")
    finally:
        if server.poll() is None:
            server.kill()
        log_file.close()


if __name__ == "__main__":
    main()
