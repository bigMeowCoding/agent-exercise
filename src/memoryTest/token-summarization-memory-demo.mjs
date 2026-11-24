import { ChatOpenAI } from '@langchain/openai';
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  getBufferString,
} from '@langchain/core/messages';
import { getEncoding } from 'js-tiktoken';

const model = new ChatOpenAI({
  modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
  apiKey: process.env.MOONSHOT_API_KEY,
  temperature: 1, // kimi-k2.6 等模型只允许 1
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL ?? 'https://api.moonshot.cn/v1',
  },
});

// 计算消息数组的总 token 数量
function countTokens(messages, encoder) {
  let total = 0;
  for (const msg of messages) {
    const content =
      typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    total += encoder.encode(content).length;
  }
  return total;
}

// ========== 总结策略演示（基于 token 计数） ==========
async function summarizationMemoryDemo() {
  const history = new InMemoryChatMessageHistory();
  const maxTokens = 200; // 超过 200 个 token 时触发总结
  const keepRecentTokens = 80; // 保留最近消息的 token 数量（约占总数的 40%）
  const enc = getEncoding('cl100k_base');

  // 模拟一段较长的对话（红烧肉教学）
  const messages = [
    { type: 'human', content: '我想学做红烧肉，你能教我吗？' },
    {
      type: 'ai',
      content:
        '当然可以！红烧肉是一道经典的家常菜。首先准备五花肉500克、冰糖、生抽、老抽、料酒、葱姜蒜。',
    },
    { type: 'human', content: '五花肉需要切多大块？' },
    {
      type: 'ai',
      content: '建议切成3-4厘米见方的块，这样既能入味又不会太碎。',
    },
    { type: 'human', content: '炒糖色的时候有什么技巧吗？' },
    {
      type: 'ai',
      content: '炒糖色要用小火，等糖融化冒小泡后再下肉，等糖色变成焦糖色。',
    },
    { type: 'human', content: '需要炖多长时间？' },
    { type: 'ai', content: '小火慢炖40-60分钟，直到肉质软烂。' },
    { type: 'human', content: '最后收汁要注意什么？' },
    { type: 'ai', content: '大火收汁时要不断翻动，防止糊底。' },
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
  const totalTokens = countTokens(allMessages, enc);

  console.log(`原始消息数量: ${allMessages.length}`);
  console.log(`原始 token 数: ${totalTokens}/${maxTokens}`);

  // 如果 token 数超过阈值，触发总结
  if (totalTokens >= maxTokens) {
    // 从后往前累加消息，保留最近的消息直到达到 keepRecentTokens
    const recentMessages = [];
    let recentTokens = 0;

    for (let i = allMessages.length - 1; i >= 0; i--) {
      const msg = allMessages[i];
      const content =
        typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
      const msgTokens = enc.encode(content).length;

      if (recentTokens + msgTokens <= keepRecentTokens) {
        recentMessages.unshift(msg);
        recentTokens += msgTokens;
      } else {
        break;
      }
    }

    const messagesToSummarize = allMessages.slice(
      0,
      allMessages.length - recentMessages.length,
    );
    const summarizeTokens = countTokens(messagesToSummarize, enc);

    console.log('\n📊 Token 数量超过阈值，开始总结...');
    console.log(
      `  将被总结的消息数量: ${messagesToSummarize.length} (${summarizeTokens} tokens)`,
    );
    console.log(
      `  将被保留的消息数量: ${recentMessages.length} (${recentTokens} tokens)`,
    );

    const summary = await summarizeHistory(messagesToSummarize);

    // 清空历史，写入摘要 + 最近消息
    await history.clear();
    await history.addMessage(new SystemMessage(`对话摘要: ${summary}`));
    for (const msg of recentMessages) {
      await history.addMessage(msg);
    }

    allMessages = await history.getMessages();
    console.log('\n保留的消息:');
    recentMessages.forEach((m) => {
      const content =
        typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
      console.log(
        `  ${m.constructor.name} (${enc.encode(content).length} tokens): ${content}`,
      );
    });
    console.log(`\n总结内容: ${summary}`);
    console.log(`\n压缩后消息数量: ${allMessages.length}`);
    console.log(`压缩后 token 数: ${countTokens(allMessages, enc)}`);
  } else {
    console.log(
      `Token 数量 (${totalTokens}) 未超过阈值 (${maxTokens})，无需总结`,
    );
  }
}

summarizationMemoryDemo().catch(console.error);

// 总结历史对话的函数
async function summarizeHistory(messages) {
  if (messages.length === 0) return '';

  const conversationText = getBufferString(messages, '用户', '助手');

  const summaryPrompt = `请总结以下对话的核心内容，保留重要信息：

${conversationText}

总结：`;

  const summaryResponse = await model.invoke([new SystemMessage(summaryPrompt)]);
  return summaryResponse.content;
}
