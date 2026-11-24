import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings } from '@langchain/openai';
import { EPubLoader } from '@langchain/community/document_loaders/fs/epub';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const COLLECTION_NAME = 'ebook_chunks';
const VECTOR_DIM = 1024;
const CHUNK_SIZE = 500;
const MILVUS_ADDRESS = process.env.MILVUS_ADDRESS ?? 'localhost:19530';
const EPUB_FILE = process.env.EPUB_FILE ?? path.join(ROOT_DIR, 'resource', '天龙八部.epub');
const BOOK_NAME = process.env.BOOK_NAME ?? '天龙八部';
// 天龙八部章节很多，默认只处理前 3 章；设为 0 表示处理全部
const MAX_CHAPTERS = Number(process.env.MAX_CHAPTERS ?? 3);

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

const client = new MilvusClient({ address: MILVUS_ADDRESS });

async function getEmbedding(text) {
    return embeddings.embedQuery(text);
}

async function ensureCollection() {
    try {
        // 检查集合是否存在
        const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });

        if (!hasCollection.value) {
            console.log('创建集合...');
            await client.createCollection({
                collection_name: COLLECTION_NAME,
                fields: [
                    { name: 'id', data_type: DataType.VarChar, max_length: 100, is_primary_key: true },
                    { name: 'book_id', data_type: DataType.VarChar, max_length: 100 },
                    { name: 'book_name', data_type: DataType.VarChar, max_length: 200 },
                    { name: 'chapter_num', data_type: DataType.Int32 },
                    { name: 'index', data_type: DataType.Int32 },
                    { name: 'content', data_type: DataType.VarChar, max_length: 10000 },
                    { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },
                ],
            });
            console.log('✓ 集合创建成功');

            // 创建索引
            await client.createIndex({
                collection_name: COLLECTION_NAME,
                field_name: 'vector',
                index_type: IndexType.IVF_FLAT,
                metric_type: MetricType.COSINE,
                params: { nlist: 1024 },
            });
            console.log('✓ 索引创建成功');
        } else {
            console.log(`✓ 集合 ${COLLECTION_NAME} 已存在`);
        }

        // 确保集合已加载
        try {
            await client.loadCollection({ collection_name: COLLECTION_NAME });
            console.log('✓ 集合已加载');
        } catch {
            console.log('✓ 集合已处于加载状态');
        }
    } catch (error) {
        console.error('创建集合时出错:', error.message);
        throw error;
    }
}

async function insertChunksBatch(chunks, bookId, chapterNum) {
    try {
        if (chunks.length === 0) {
            return 0;
        }

        // 为每个文档块生成向量并构建插入数据
        const insertData = await Promise.all(
            chunks.map(async (chunk, chunkIndex) => {
                const vector = await getEmbedding(chunk);
                // 手动生成 ID: book_id_chapterNum_index
                return {
                    id: `${bookId}_${chapterNum}_${chunkIndex}`,
                    book_id: String(bookId),
                    book_name: BOOK_NAME,
                    chapter_num: chapterNum,
                    index: chunkIndex,
                    content: chunk.slice(0, 10000),
                    vector,
                };
            })
        );

        // 批量插入到 Milvus
        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: insertData,
        });

        return Number(insertResult.insert_cnt) || 0;
    } catch (error) {
        console.error(`插入第 ${chapterNum} 章时出错:`, error.message);
        throw error;
    }
}

async function loadAndProcessEPubStreaming(bookId) {
    try {
        console.log(`\n开始加载 EPUB 文件: ${EPUB_FILE}`);

        // 使用 EPubLoader 加载文件，按章节拆分
        const loader = new EPubLoader(EPUB_FILE, {
            splitChapters: true,
        });

        const documents = await loader.load();
        console.log(`✓ 加载完成，共 ${documents.length} 个章节\n`);

        // 创建文本拆分器，拆分到 500 个字符
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: CHUNK_SIZE,
            chunkOverlap: 50, // 重叠 50 个字符，保持上下文连贯性
        });

        let totalInserted = 0;
        const chapterLimit =
            MAX_CHAPTERS > 0 ? Math.min(documents.length, MAX_CHAPTERS) : documents.length;

        if (MAX_CHAPTERS > 0 && documents.length > MAX_CHAPTERS) {
            console.log(`演示模式：只处理前 ${MAX_CHAPTERS} 章（设 MAX_CHAPTERS=0 可处理全部）\n`);
        }

        // 遍历每个章节，进行二次拆分并立即插入
        for (let chapterIndex = 0; chapterIndex < chapterLimit; chapterIndex++) {
            const chapter = documents[chapterIndex];
            const chapterContent = chapter.pageContent;
            const chapterTitle = chapter.metadata?.chapter ?? `第 ${chapterIndex + 1} 章`;

            console.log(`处理第 ${chapterIndex + 1}/${chapterLimit} 章: ${chapterTitle}`);

            // 使用 splitter 进行二次拆分
            const chunks = await textSplitter.splitText(chapterContent);

            console.log(`  拆分为 ${chunks.length} 个片段`);

            if (chunks.length === 0) {
                console.log('  跳过空章节\n');
                continue;
            }

            console.log('  生成向量并插入中...');

            // 立即生成向量并插入该章节的所有片段
            const insertedCount = await insertChunksBatch(chunks, bookId, chapterIndex + 1);
            totalInserted += insertedCount;
            console.log(`  ✓ 已插入 ${insertedCount} 条\n`);
        }

        console.log(`合计插入 ${totalInserted} 条记录`);
        return totalInserted;
    } catch (error) {
        console.error('处理 EPUB 时出错:', error.message);
        throw error;
    }
}

async function main() {
    try {
        console.log('='.repeat(80));
        console.log('电子书处理程序');
        console.log('='.repeat(80));

        // 连接 Milvus
        console.log('\n连接 Milvus...');
        await client.connectPromise;
        console.log('✓ 已连接\n');

        // 设置 book_id
        const bookId = 1;

        // 确保集合存在
        await ensureCollection(bookId);

        // 加载和处理 EPUB 文件（流式处理，边处理边插入）
        await loadAndProcessEPubStreaming(bookId);

        console.log('='.repeat(80));
        console.log('处理完成!');
        console.log('='.repeat(80));
    } catch (error) {
        console.error('\n错误:', error.message);
        console.error(error.stack);
        process.exitCode = 1;
    } finally {
        await client.closeConnection();
    }
}

main();
