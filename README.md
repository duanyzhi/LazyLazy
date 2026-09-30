# LazyLazy

极简 LLM 推理框架（vLLM 风格），支持 Qwen3 系列模型。

## 环境准备

## 启动 API 服务

```bash
python -m lazy.api_server --model /mnt/models/Qwen3-0.6B-Base --port 8000
```

看到 `serving ... on http://0.0.0.0:8000` 说明服务就绪（0.6B 模型加载大概 10 秒）。

参数说明：

- `--model`：本地模型路径（必填）
- `--host`：监听地址，默认 `0.0.0.0`
- `--port`：端口，默认 `8000`

## 测试

健康检查：

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

文本补全 `/v1/completions`：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"what is the largest animal in the world?","max_tokens":32,"temperature":0.001}'
```

对话 `/v1/chat/completions`：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi, who are you?"}],"max_tokens":32,"temperature":0.001}'
```

返回是 OpenAI 风格的 JSON，包含 `choices`、`usage`（token 统计）、`finish_reason` 字段。

注意事项：

- `temperature` 必须大于 0（引擎不支持贪心采样），想要接近确定性输出传 `0.001`
- 想看每步 decode 的详细日志，启动前加环境变量：`LAZY_LOG_LEVEL=DEBUG python -m lazy.api_server ...`
- 引擎当前单请求串行处理，并发请求会自动排队

## 本地 Python 调用（非服务方式）

```bash
cd tests
python llm_test.py
```
