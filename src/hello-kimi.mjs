import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI({
    modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
    apiKey: process.env.MOONSHOT_API_KEY,
    configuration: {
        baseURL: 'https://api.moonshot.cn/v1',
    },
});

const response = await model.invoke('介绍下自己');
console.log(response.content);
