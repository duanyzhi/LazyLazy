import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lazy import LLM, SamplingParams
from lazy.utils.logger import init_logger

logger = init_logger("lazy.api_server")  # python -m 运行时 __name__ 是 __main__，用固定名字让日志显示真实模块名


class LazyAPIServer:
    """把 LLM 包成 OpenAI 风格的 HTTP 服务（MVP）。

    引擎当前只保证单请求安全（KVCache 全局一份、generate 结束会 reset），
    所以所有生成请求用一把锁串行处理，并发请求排队等待。
    """

    def __init__(self, model: str):
        self.llm = LLM(model)
        self.model = model
        self.lock = threading.Lock()

    def chat_prompt(self, messages: list) -> str:
        tokenizer = self.llm.tokenizer
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("model has no chat template, use /v1/completions instead")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_text(self, prompt: str, max_tokens: int, temperature: float):
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        num_prompt_tokens = len(self.llm.tokenizer.encode(prompt))
        logger.debug("waiting for engine lock")
        with self.lock:
            text = self.llm.generate(prompt, params)[0]
        num_completion_tokens = len(self.llm.tokenizer.encode(text))
        return text, num_prompt_tokens, num_completion_tokens


def make_handler(api: LazyAPIServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug("http: " + format, *args)

        def _send(self, code: int, obj: dict):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            try:
                body = self._read_json()
            except Exception:
                return self._send(400, {"error": "invalid json body"})
            if self.path == "/v1/completions":
                prompt = body.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    return self._send(400, {"error": "prompt must be a non-empty string"})
                self._generate(body, prompt, chat=False)
            elif self.path == "/v1/chat/completions":
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    return self._send(400, {"error": "messages must be a non-empty list"})
                try:
                    prompt = api.chat_prompt(messages)
                except ValueError as err:
                    return self._send(400, {"error": str(err)})
                self._generate(body, prompt, chat=True)
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def _generate(self, body: dict, prompt: str, chat: bool):
            max_tokens = int(body.get("max_tokens", 64))
            temperature = float(body.get("temperature", 1.0))
            if temperature <= 1e-10:
                return self._send(400, {"error": "temperature must be > 1e-10, greedy sampling is not permitted"})
            logger.info("request received: %s, max_tokens=%d, temperature=%g", self.path, max_tokens, temperature)
            try:
                text, num_prompt, num_completion = api.generate_text(prompt, max_tokens, temperature)
            except Exception as err:
                logger.exception("generation failed")
                return self._send(500, {"error": f"generation failed: {err}"})

            finish_reason = "length" if num_completion >= max_tokens else "stop"
            logger.info("request done: %s, %d prompt tokens -> %d completion tokens (%s)",
                        self.path, num_prompt, num_completion, finish_reason)
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text},
                          "finish_reason": finish_reason}
                obj = "chat.completion"
            else:
                choice = {"index": 0, "text": text, "finish_reason": finish_reason}
                obj = "text_completion"
            self._send(200, {
                "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": obj,
                "created": int(time.time()),
                "model": api.model,
                "choices": [choice],
                "usage": {"prompt_tokens": num_prompt,
                          "completion_tokens": num_completion,
                          "total_tokens": num_prompt + num_completion},
            })

    return Handler


def main():
    parser = argparse.ArgumentParser(description="LazyLazy OpenAI-style API server (MVP)")
    parser.add_argument("--model", required=True, help="local model path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api = LazyAPIServer(args.model)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(api))
    logger.info("serving %s on http://%s:%d", args.model, args.host, args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
