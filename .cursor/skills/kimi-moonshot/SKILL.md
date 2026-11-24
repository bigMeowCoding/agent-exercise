---
name: kimi-moonshot
description: >-
  Configures and debugs Kimi/Moonshot ChatOpenAI calls in this repo (LangChain,
  temperature, env, baseURL, embeddings). Use when adding LLM demos, fixing
  Moonshot 400 errors, invalid temperature, model 404, or MOONSHOT_API_KEY /
  KIMI_MODEL issues.
---

# Kimi / Moonshot 调用约定

本仓库通过 OpenAI 兼容接口调用 Kimi（Moonshot）。写或改任何 `ChatOpenAI` 示例时遵循下列约定。

## 标准初始化

```js
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
  modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
  apiKey: process.env.MOONSHOT_API_KEY,
  temperature: 1, // kimi-k2.6 等模型只允许 1
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL ?? 'https://api.moonshot.cn/v1',
  },
});
```

## 硬性约束

| 项 | 约定 |
|---|---|
| 默认模型 | `kimi-k2.6`（可用 `KIMI_MODEL` 覆盖） |
| API Key | `MOONSHOT_API_KEY`（见 `.env.example`） |
| baseURL | `https://api.moonshot.cn/v1` |
| temperature | **必须为 `1`**；不要传 `0` 或其他值 |
| Embedding | Moonshot **不提供**；用 `EMBEDDINGS_*` 配独立 OpenAI 兼容服务 |

## 常见报错

### `invalid temperature: only 1 is allowed for this model`

原因：给模型传了非 `1` 的 temperature（例如 `0`）。

处理：改为 `temperature: 1`，或省略该字段且确认客户端不会注入其他默认值。本仓库约定显式写 `1`。

### 模型 404 / model not found

原因：`KIMI_MODEL` 或默认名在当前账号不可用。

处理：查可用模型 `GET https://api.moonshot.cn/v1/models`，用 `KIMI_MODEL` 改成可用名称。

### 缺 Key / Embedding 失败

- Chat：缺 `MOONSHOT_API_KEY`
- RAG Embedding：缺 `EMBEDDINGS_API_KEY` / `EMBEDDINGS_BASE_URL` / `EMBEDDINGS_MODEL_NAME`（参考 `.env.example`）

## 新增示例检查清单

- [ ] 使用上述标准 `ChatOpenAI` 初始化
- [ ] `temperature` 为 `1`（未写 `0`）
- [ ] 通过 `npm` script + `--env-file-if-exists=.env` 启动
- [ ] 不把 `.env` 提交进 Git
- [ ] 需要向量时单独配置 Embedding，不走 Moonshot
