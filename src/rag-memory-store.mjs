import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';
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

const documents = [
    new Document({
        pageContent: '光光是一个活泼开朗的小男孩，他有一双明亮的大眼睛，总是带着灿烂的笑容。光光最喜欢的事情就是踢足球，每天放学后都会在操场上奔跑。',
        metadata: { chapter: 1, character: '光光', type: '角色介绍', mood: '活泼' },
    }),
    new Document({
        pageContent: '东东是光光最好的朋友，他是一个安静而聪明的男孩。东东喜欢读书和画画，他的画总是充满了奇妙的想象，同学们都很喜欢看。',
        metadata: { chapter: 2, character: '东东', type: '角色介绍', mood: '温馨' },
    }),
    new Document({
        pageContent: '有一天，学校要举办一场足球比赛，光光非常兴奋，他邀请东东一起参加。但是东东从来没有踢过足球，他担心自己会拖大家的后腿。光光拍拍他的肩膀说："别怕，我教你！"',
        metadata: { chapter: 3, character: '光光和东东', type: '友情情节', mood: '鼓励' },
    }),
    new Document({
        pageContent: '接下来的日子里，光光每天放学后都会教东东踢足球。光光耐心地教东东如何控球、传球和射门，东东摔倒了好几次，但每次都在光光的鼓励下重新站了起来。',
        metadata: { chapter: 4, character: '光光和东东', type: '友情情节', mood: '互助' },
    }),
    new Document({
        pageContent: '比赛那天终于到了，光光和东东一起站在球场上。虽然东东的技术还不够熟练，但他非常努力，在最后一分钟把球传给了光光，光光一脚射门，球队赢得了比赛！',
        metadata: { chapter: 5, character: '光光和东东', type: '高潮转折', mood: '激动' },
    }),
    new Document({
        pageContent: '从那以后，光光和东东成为了学校里最要好的朋友。光光教东东运动，东东教光光画画，他们互相学习，一起成长，每天都过得很开心。',
        metadata: { chapter: 6, character: '光光和东东', type: '结局', mood: '欢乐' },
    }),
    new Document({
        pageContent: '多年后，光光成为了一名职业足球运动员，而东东成为了一名优秀的插画师。虽然他们走上了不同的道路，但每年夏天都会回到当年的操场，一起踢一场球。',
        metadata: { chapter: 7, character: '光光和东东', type: '尾声', mood: '温馨' },
    }),
];

const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);

const questions = ['东东和光光是怎么成为朋友的？'];

for (const question of questions) {
    console.log('='.repeat(80));
    console.log(`问题: ${question}`);
    console.log('='.repeat(80));

    // MemoryVectorStore 返回的分数是余弦相似度，越大越相似
    const scoredResults = await vectorStore.similaritySearchWithScore(question, 5);

    console.log('\n【检索到的文档及相似度】');
    scoredResults.forEach(([doc, score], i) => {
        console.log(`\n[文档 ${i + 1}] 相似度: ${score.toFixed(4)}`);
        console.log(`内容: ${doc.pageContent}`);
        console.log(`元数据: 章节=${doc.metadata.chapter}, 角色=${doc.metadata.character}, 类型=${doc.metadata.type}, 情绪=${doc.metadata.mood}`);
    });

    const context = scoredResults
        .map(([doc], i) => `[片段${i + 1}]\n${doc.pageContent}`)
        .join('\n\n━━━━━\n\n');

    const prompt = `你是一个讲友情故事的老师。基于以下故事片段回答问题，用温暖生动的语言。如果故事中没有提到相关信息，就诚实地说不知道。

故事片段:
${context}

问题: ${question}

老师的回答:`;

    console.log('\n【AI 回答】');
    const response = await model.invoke(prompt);
    console.log(response.content);
    console.log('\n');
}
