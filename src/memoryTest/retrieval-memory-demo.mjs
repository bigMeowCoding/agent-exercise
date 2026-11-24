import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { MilvusClient, MetricType } from '@zilliz/milvus2-sdk-node';
import { HumanMessage } from '@langchain/core/messages';

const COLLECTION_NAME = 'conversations';
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? 'localhost:19530';

const missingChatEnv = ['MOONSHOT_API_KEY'].filter((name) => !process.env[name]);
const missingEmbedEnv = ['EMBEDDINGS_API_KEY', 'EMBEDDINGS_BASE_URL', 'EMBEDDINGS_MODEL_NAME'].filter(
  (name) => !process.env[name],
);
const missingEnv = [...missingChatEnv, ...missingEmbedEnv];
if (missingEnv.length) {
  console.error(`缺少环境变量: ${missingEnv.join(', ')}，请参考 .env.example 在 .env 中配置`);
  process.exit(1);
}

// 初始化 Chat 模型
const model = new ChatOpenAI({
  modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
  apiKey: process.env.MOONSHOT_API_KEY,
  temperature: 1, // kimi-k2.6 等模型只允许 1
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL ?? 'https://api.moonshot.cn/v1',
  },
});

// 初始化 Embeddings 模型
const embeddings = new OpenAIEmbeddings({
  model: process.env.EMBEDDINGS_MODEL_NAME,
  apiKey: process.env.EMBEDDINGS_API_KEY,
  // OpenAI SDK 默认请求 base64，部分兼容服务不支持
  encodingFormat: 'float',
  configuration: {
    baseURL: process.env.EMBEDDINGS_BASE_URL,
  },
});

// 初始化 Milvus 客户端
const client = new MilvusClient({ address: MILVUS_ADDRESS });

/** 获取文本的向量嵌入 */
async function getEmbedding(text) {
  return embeddings.embedQuery(text);
}

/**
 * 从 Milvus 中检索相关的历史对话
 */
async function retrieveRelevantConversations(query, k = 2) {
  try {
    // 生成查询的向量
    const queryVector = await getEmbedding(query);

    // 在 Milvus 中搜索相似的对话
    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      vector: queryVector,
      limit: k,
      metric_type: MetricType.COSINE,
      output_fields: ['id', 'content', 'round', 'timestamp'],
    });

    return searchResult.results;
  } catch (error) {
    console.error('检索对话时出错:', error.message);
    return [];
  }
}

/**
 * 策略3: 检索 (Retrieval)
 * 使用 Milvus 向量数据库存储历史对话，根据当前输入检索语义相关的历史
 * 实现 RAG (Retrieval-Augmented Generation) 流程
 */
async function retrievalMemoryDemo() {
  try {
    console.log('连接到 Milvus...');
    await client.connectPromise;
    console.log('✓ 已连接');
  } catch (error) {
    console.error('无法连接到 Milvus:', error.message);
    console.error('请确认本机已启动 Milvus 服务（默认 localhost:19530）');
    return;
  }

  try {
    // 创建历史消息存储
    const history = new InMemoryChatMessageHistory();

    const conversations = [
      { input: '我之前提到的机器学习项目进展如何？' },
      { input: '我周末经常做什么？' },
      { input: '我的职业是什么？' },
    ];

    for (let i = 0; i < conversations.length; i++) {
      const { input } = conversations[i];
      const userMessage = new HumanMessage(input);

      console.log(`\n${'='.repeat(60)}`);
      console.log(`[第 ${i + 1} 轮对话]`);
      console.log(`用户: ${input}`);

      // 1. 检索相关的历史对话
      const retrievedConversations = await retrieveRelevantConversations(input, 2);

      // 2. 将检索结果拼成上下文
      let relevantHistory = '';
      if (retrievedConversations.length > 0) {
        retrievedConversations.forEach((conv, idx) => {
          console.log(`\n[历史对话 ${idx + 1}] 相似度: ${conv.score.toFixed(4)}`);
          console.log(`轮次: ${conv.round}`);
          console.log(`内容: ${conv.content}`);
        });

        relevantHistory = retrievedConversations
          .map((conv, idx) => {
            return `[历史对话 ${idx + 1}]
轮次: ${conv.round}
${conv.content}`;
          })
          .join('\n\n——\n\n');
      } else {
        console.log('未找到相关历史对话');
      }

      const contextMessages = relevantHistory
        ? [
            new HumanMessage(
              `相关历史对话:\n${relevantHistory}\n\n用户问题: ${input}`,
            ),
          ]
        : [userMessage];

      // 3. 调用模型生成回答
      console.log('\n[AI 回答]');
      const response = await model.invoke(contextMessages);

      // 保存当前对话历史消息
      await history.addMessage(userMessage);
      await history.addMessage(response);

      // 4. 将对话保存到 Milvus 向量数据库
      const conversationText = `用户: ${input}\n助手: ${response.content}`;
      const convId = `conv_${Date.now()}_${i + 1}`;
      const convVector = await getEmbedding(conversationText);

      try {
        await client.insert({
          collection_name: COLLECTION_NAME,
          data: [
            {
              id: convId,
              vector: convVector,
              content: conversationText,
              round: i + 1,
              timestamp: new Date().toISOString(),
            },
          ],
        });
        console.log('💾 已保存到 Milvus 向量数据库');
      } catch (error) {
        console.warn('保存到向量数据库时出错:', error.message);
      }

      console.log(`助手: ${response.content}`);
    }
  } finally {
    await client.closeConnection();
  }
}

retrievalMemoryDemo().catch(console.error);
