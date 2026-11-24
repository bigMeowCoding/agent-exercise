import { ChatOpenAI } from '@langchain/openai';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  getBufferString,
} from '@langchain/core/messages';

const model = new ChatOpenAI({
  modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
  apiKey: process.env.MOONSHOT_API_KEY,
  temperature: 1,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL ?? 'https://api.moonshot.cn/v1',
  },
});

// =========== 总结策略演示 ===========
async function summarizationMemoryDemo() {
  const history = new InMemoryChatMessageHistory();
  const maxMessages = 6; // 超过 6 条消息时触发总结

  // 模拟一段较长的对话（红烧肉教学）
  const messages = [
    { type: 'human', content: '我想学做红烧肉，你能教我吗？' },
    {
      type: 'ai',
      content:
        '当然可以！红烧肉是一道经典的家常菜。首先准备五花肉500克、冰糖、生抽、老抽、料酒、葱姜蒜。',
    },
    { type: 'human', content: '五花肉需要切多大块？' },
    { type: 'ai', content: '建议切成3-4厘米见方的块，这样既能入味又不会太碎。' },
    { type: 'human', content: '炒糖色的时候有什么技巧吗？' },
    {
      type: 'ai',
      content: '炒糖色要用小火，等糖融化冒小泡后再下肉，这样颜色更好看。',
    },
    { type: 'human', content: '需要炖多长时间？' },
    {
      type: 'ai',
      content: '小火慢炖40-50分钟，直到肉质软烂，汤汁浓稠即可。',
    },
    { type: 'human', content: '最后收汁要注意什么？' },
    {
      type: 'ai',
      content: '大火收汁时要不断翻动，防止糊底，汤汁裹匀每块肉就好。',
    },
  ];

  // 添加所有消息
  for (const msg of messages) {
    if (msg.type === 'human') {
      await history.addMessage(new HumanMessage(msg.content));
    } else {
      await history.addMessage(new AIMessage(msg.content));
    }
  }

  let allMessages = await history.getMessages();
  console.log(`原始消息数量: ${allMessages.length}`);
  console.log(
    '原始消息:',
    allMessages.map((m) => `${m.constructor.name}: ${m.content}`).join('\n'),
  );

  // 如果消息过多，触发总结
  if (allMessages.length >= maxMessages) {
    console.log('\n消息过多，开始总结...');
    const keepRecent = 2; // 保留最近 2 条消息
    const recentMessages = allMessages.slice(-keepRecent);
    const messagesToSummarize = allMessages.slice(0, -keepRecent);

    console.log(`总结 ${messagesToSummarize.length} 条消息，保留最近 ${keepRecent} 条`);
    console.log(
      '保留的消息:',
      recentMessages.map((m) => `${m.constructor.name}: ${m.content}`).join('\n'),
    );

    const summary = await summarizeHistory(messagesToSummarize);
    console.log(`\n总结内容: ${summary}`);

    // 清空历史，写入摘要 + 最近消息
    await history.clear();
    await history.addMessage(new SystemMessage(`对话摘要: ${summary}`));
    for (const msg of recentMessages) {
      await history.addMessage(msg);
    }

    allMessages = await history.getMessages();
    console.log(`\n压缩后消息数量: ${allMessages.length}`);
    console.log(
      '压缩后消息:',
      allMessages.map((m) => `${m.constructor.name}: ${m.content}`).join('\n'),
    );
  } else {
    console.log('消息数量未超过阈值，无需总结');
  }
}

summarizationMemoryDemo().catch(console.error);

// 总结历史对话的函数
async function summarizeHistory(messages) {
  if (messages.length === 0) return '';

  const conversationText = getBufferString(messages, '用户', '助手');

  const summaryPrompt = `请总结以下对话的核心内容，保留重要信息：\n\n${conversationText}\n\n总结：`;

  const summaryResponse = await model.invoke([new SystemMessage(summaryPrompt)]);
  return summaryResponse.content;
}
