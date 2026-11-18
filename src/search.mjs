import { MilvusClient, MetricType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from '@langchain/openai';

const COLLECTION_NAME = 'ai_diary';
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? 'localhost:19530';

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
    // OpenAI SDK 默认请求 base64，部分兼容服务不支持
    encodingFormat: 'float',
    configuration: {
        baseURL: process.env.EMBEDDINGS_BASE_URL,
    },
});

const client = new MilvusClient({ address: MILVUS_ADDRESS });

async function getEmbedding(text) {
    return embeddings.embedQuery(text);
}

async function main() {
    try {
        console.log('Connecting to Milvus...');
        await client.connectPromise;
        console.log('✓ Connected\n');

        // 向量搜索
        console.log('Searching for similar diary entries...');
        const query = '我想看看关于户外活动的日记';
        console.log(`Query: "${query}"\n`);

        const queryVector = await getEmbedding(query);

        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: 2,
            metric_type: MetricType.COSINE,
            output_fields: ['id', 'content', 'date', 'mood', 'tags'],
        });

        console.log(`Found ${searchResult.results.length} results:\n`);

        searchResult.results.forEach((item, index) => {
            console.log(`${index + 1}. [Score: ${item.score.toFixed(4)}]`);
            console.log(`   ID: ${item.id}`);
            console.log(`   Date: ${item.date}`);
            console.log(`   Mood: ${item.mood}`);
            console.log(`   Tags: ${item.tags?.join(', ')}`);
            console.log(`   Content: ${item.content}\n`);
        });
    } catch (error) {
        console.error('Error:', error.message);
        process.exitCode = 1;
    } finally {
        await client.closeConnection();
    }
}

main();
