import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from '@langchain/openai';

const COLLECTION_NAME = 'ai_diary';
const VECTOR_DIM = 1024;
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
        console.log('Connected');

        const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
        if (hasCollection.value) {
            console.log(`\nCollection ${COLLECTION_NAME} already exists, dropping...`);
            await client.dropCollection({ collection_name: COLLECTION_NAME });
        }

        // 创建集合
        console.log('\nCreating collection...');
        await client.createCollection({
            collection_name: COLLECTION_NAME,
            fields: [
                { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
                { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },
                { name: 'content', data_type: DataType.VarChar, max_length: 5000 },
                { name: 'date', data_type: DataType.VarChar, max_length: 50 },
                { name: 'mood', data_type: DataType.VarChar, max_length: 50 },
                {
                    name: 'tags',
                    data_type: DataType.Array,
                    element_type: DataType.VarChar,
                    max_capacity: 10,
                    max_length: 50,
                },
            ],
        });
        console.log('Collection created');

        // 创建索引
        console.log('\nCreating index...');
        await client.createIndex({
            collection_name: COLLECTION_NAME,
            field_name: 'vector',
            index_type: IndexType.IVF_FLAT,
            metric_type: MetricType.COSINE,
            params: { nlist: 128 },
        });
        console.log('Index created');

        // 加载集合
        console.log('\nLoading collection...');
        await client.loadCollection({ collection_name: COLLECTION_NAME });
        console.log('Collection loaded');

        // 插入日记数据
        console.log('\nInserting diary entries...');
        const diaryContents = [
            {
                id: 'diary_001',
                content: '今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。',
                date: '2026-01-10',
                mood: 'happy',
                tags: ['生活', '散步'],
            },
            {
                id: 'diary_002',
                content: '今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。',
                date: '2026-01-11',
                mood: 'excited',
                tags: ['工作', '成就'],
            },
            {
                id: 'diary_003',
                content: '周末和朋友一起去爬山，虽然很累但是风景很美。在山顶看日落，感觉人生值得。',
                date: '2026-01-12',
                mood: 'relaxed',
                tags: ['户外', '朋友'],
            },
            {
                id: 'diary_004',
                content: '今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。',
                date: '2026-01-13',
                mood: 'curious',
                tags: ['学习', '技术'],
            },
            {
                id: 'diary_005',
                content: '做了一顿丰盛的晚饭，家人都很喜欢。一起吃饭聊天的时光最温馨。',
                date: '2026-01-13',
                mood: 'proud',
                tags: ['美食', '家庭'],
            },
        ];

        console.log('Generating embeddings...');
        const diaryData = await Promise.all(
            diaryContents.map(async (diary) => ({
                ...diary,
                vector: await getEmbedding(diary.content),
            }))
        );

        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: diaryData,
        });
        console.log(`✔ Inserted ${insertResult.insert_cnt} records\n`);
    } catch (error) {
        console.error('Error:', error.message);
        process.exitCode = 1;
    } finally {
        await client.closeConnection();
    }
}

main();
