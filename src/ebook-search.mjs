import { MilvusClient, MetricType } from '@zilliz/milvus2-sdk-node';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';

const COLLECTION_NAME = 'ebook_chunks';
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? 'localhost:19530';
const QUERY = process.env.EBOOK_QUERY ?? '乔峰为什么被逐出丐帮？';

const missingEnv = ['EMBEDDINGS_API_KEY', 'EMBEDDINGS_BASE_URL', 'EMBEDDINGS_MODEL_NAME'].filter(
    (name) => !process.env[name]
);
if (missingEnv.length) {
    console.error(`缺少环境变量: ${missingEnv.join(', ')}，请参考 .env.example 在 .env 中配置`);
    process.exit(1);
}

const embeddings = new OpenAIEmbeddings({
    model: process.env.EMBEDDINGS_MODEL_NAME,
    apiKey: process.env.EMBEDDINGS_API_KEY,
    encodingFormat: 'float',
    configuration: {
        baseURL: process.env.EMBEDDINGS_BASE_URL,
    },
});

const model = process.env.MOONSHOT_API_KEY
    ? new ChatOpenAI({
          modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
          apiKey: process.env.MOONSHOT_API_KEY,
          configuration: {
              baseURL: 'https://api.moonshot.cn/v1',
          },
      })
    : null;

const client = new MilvusClient({ address: MILVUS_ADDRESS });

async function getEmbedding(text) {
    return embeddings.embedQuery(text);
}

async function main() {
    try {
        console.log('='.repeat(80));
        console.log('电子书检索 Demo');
        console.log('='.repeat(80));

        console.log('\n连接 Milvus...');
        await client.connectPromise;
        console.log('✓ 已连接\n');

        console.log(`问题: ${QUERY}\n`);
        const queryVector = await getEmbedding(QUERY);

        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: 3,
            metric_type: MetricType.COSINE,
            output_fields: ['id', 'book_name', 'chapter_num', 'index', 'content'],
        });

        console.log(`检索到 ${searchResult.results.length} 条相关片段:\n`);

        searchResult.results.forEach((item, i) => {
            console.log(`${i + 1}. [Score: ${item.score.toFixed(4)}] 《${item.book_name}》第 ${item.chapter_num} 章 #${item.index}`);
            console.log(`   ${item.content}\n`);
        });

        if (model && searchResult.results.length) {
            const context = searchResult.results
                .map(
                    (item, i) =>
                        `[片段${i + 1}] 第${item.chapter_num}章\n${item.content}`
                )
                .join('\n\n━━━━━\n\n');

            const prompt = `你是一个电子书阅读助手。根据检索到的书本片段回答问题。如果片段中没有相关信息，就诚实地说不知道。

书本片段:
${context}

问题: ${QUERY}

你的回答:`;

            console.log('【AI 回答】');
            const response = await model.invoke(prompt);
            console.log(response.content);
            console.log();
        }
    } catch (error) {
        console.error('错误:', error.message);
        process.exitCode = 1;
    } finally {
        await client.closeConnection();
    }
}

main();
