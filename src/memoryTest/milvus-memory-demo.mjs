import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from '@langchain/openai';

const COLLECTION_NAME = 'conversations';
const VECTOR_DIM = 1024;
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? 'localhost:19530';

const missingEnv = ['EMBEDDINGS_API_KEY', 'EMBEDDINGS_BASE_URL', 'EMBEDDINGS_MODEL_NAME'].filter(
  (name) => !process.env[name],
);
if (missingEnv.length) {
  console.error(`缺少环境变量: ${missingEnv.join(', ')}，请参考 .env.example 在 .env 中配置`);
  process.exit(1);
}

const embeddings = new OpenAIEmbeddings({
  model: process.env.EMBEDDINGS_MODEL_NAME,
  apiKey: process.env.EMBEDDINGS_API_KEY,
  // OpenAI SDK 默认请求 base64，部分兼容服务不支持
  encodingFormat: 'float',
  configuration: {
    baseURL: process.env.EMBEDDINGS_BASE_URL,
  },
});

const client = new MilvusClient({ address: MILVUS_ADDRESS });

/** 获取文本的向量嵌入 */
async function getEmbedding(text) {
  const result = await embeddings.embedQuery(text);
  return result;
}

async function main() {
  try {
    console.log('连接到 Milvus...');
    await client.connectPromise;
    console.log('✓ 已连接');

    const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
    if (hasCollection.value) {
      console.log(`\n集合 ${COLLECTION_NAME} 已存在，先删除...`);
      await client.dropCollection({ collection_name: COLLECTION_NAME });
    }

    // 创建集合
    console.log('\n创建集合...');
    await client.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },
        { name: 'content', data_type: DataType.VarChar, max_length: 5000 },
        { name: 'round', data_type: DataType.Int64 },
        { name: 'timestamp', data_type: DataType.VarChar, max_length: 100 },
      ],
    });
    console.log('✓ 集合已创建');

    // 创建索引
    console.log('\n创建索引...');
    await client.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'vector',
      index_type: IndexType.IVF_FLAT,
      metric_type: MetricType.COSINE,
      params: { nlist: 128 },
    });
    console.log('✓ 索引已创建');

    // 加载集合
    console.log('\n加载集合...');
    await client.loadCollection({ collection_name: COLLECTION_NAME });
    console.log('✓ 集合已加载');

    // 插入对话数据
    console.log('\n插入对话数据...');
    const conversations = [
      {
        id: 'conv_001',
        content: '用户：我叫赵六，是一名数据科学家\n助手：很高兴认识你，赵六！数据科学是一个很有趣的领域。',
        round: 1,
        timestamp: new Date().toISOString(),
      },
      {
        id: 'conv_002',
        content: '用户：我最近在研究机器学习算法\n助手：机器学习确实很有意思，你在研究哪些算法呢？',
        round: 2,
        timestamp: new Date().toISOString(),
      },
      {
        id: 'conv_003',
        content: '用户：我喜欢打篮球和看电影\n助手：运动和文化娱乐都是很好的爱好！',
        round: 3,
        timestamp: new Date().toISOString(),
      },
      {
        id: 'conv_004',
        content: '用户：我周末经常去电影院\n助手：看电影是很好的放松方式。',
        round: 4,
        timestamp: new Date().toISOString(),
      },
      {
        id: 'conv_005',
        content: '用户：我的职业是软件工程师\n助手：软件工程师是个很有前景的职业！',
        round: 5,
        timestamp: new Date().toISOString(),
      },
    ];

    console.log('生成向量嵌入...');
    const conversationData = await Promise.all(
      conversations.map(async (conv) => ({
        ...conv,
        vector: await getEmbedding(conv.content),
      })),
    );

    const insertResult = await client.insert({
      collection_name: COLLECTION_NAME,
      data: conversationData,
    });
    console.log(`✓ 已插入 ${insertResult.insert_cnt} 条记录\n`);

    console.log('='.repeat(60));
    console.log('说明：已成功将对话数据插入到 Milvus 向量数据库');
    console.log('这些对话数据将用于后续的 RAG 检索');
    console.log('='.repeat(60) + '\n');
  } catch (error) {
    console.error('错误:', error.message);
    process.exitCode = 1;
  } finally {
    await client.closeConnection();
  }
}

main();
