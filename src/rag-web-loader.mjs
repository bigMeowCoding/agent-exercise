import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';

const model = new ChatOpenAI({
    modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
    apiKey: process.env.MOONSHOT_API_KEY,
    configuration: {
        baseURL: 'https://api.moonshot.cn/v1',
    },
});

// Moonshot 不提供 Embedding 接口，需单独配置一个 OpenAI 兼容的 Embedding 服务
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

const cheerioLoader = new CheerioWebBaseLoader('https://juejin.cn/post/7233327509919547452', {
    selector: '.main-area p',
});

const documents = await cheerioLoader.load();
console.log(`文章加载完成，共 ${documents[0].pageContent.length} 个字符`);

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
    separators: ['。', '！', '？'],
});

const splitDocuments = await textSplitter.splitDocuments(documents);
console.log(`文档分割完成，共 ${splitDocuments.length} 个分块\n`);

const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, embeddings);

const questions = ['父亲的去世对作者的人生态度产生了怎样的根本性逆转？'];

for (const question of questions) {
    console.log('='.repeat(80));
    console.log(`问题: ${question}`);
    console.log('='.repeat(80));

    // MemoryVectorStore 返回的分数是余弦相似度，越大越相似
    const scoredResults = await vectorStore.similaritySearchWithScore(question, 2);

    console.log('\n【检索到的文档及相似度】');
    scoredResults.forEach(([doc, score], i) => {
        console.log(`\n[文档 ${i + 1}] 相似度: ${score.toFixed(4)}`);
        console.log(`内容: ${doc.pageContent}`);
    });

    const context = scoredResults
        .map(([doc], i) => `[片段${i + 1}]\n${doc.pageContent}`)
        .join('\n\n━━━━━\n\n');

    const prompt = `你是一个文章辅助阅读助手，根据文章内容来解答。如果文章中没有提到相关信息，就诚实地说不知道。

文章内容:
${context}

问题: ${question}

你的回答:`;

    console.log('\n【AI 回答】');
    const response = await model.invoke(prompt);
    console.log(response.content);
    console.log('\n');
}
