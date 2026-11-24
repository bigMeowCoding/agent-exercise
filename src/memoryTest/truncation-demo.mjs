import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import { HumanMessage, AIMessage, trimMessages } from '@langchain/core/messages';
import { getEncoding } from 'js-tiktoken';

// =========== 1. 按消息数量截断 ===========
async function messageCountTruncation() {
  console.log('\n========== 按消息数量截断 ==========\n');

  const history = new InMemoryChatMessageHistory();
  const maxMessages = 4; // 只保留最近 4 条消息

  // 模拟一段较长的对话
  const messages = [
    { type: 'human', content: '我叫张三' },
    { type: 'ai', content: '你好张三，很高兴认识你！' },
    { type: 'human', content: '我今年25岁' },
    { type: 'ai', content: '25岁正是拼搏的好年纪！' },
    { type: 'human', content: '我喜欢编程' },
    { type: 'ai', content: '编程是一项很有趣的技能！你主要用什么语言？' },
    { type: 'human', content: '我住在北京' },
    { type: 'ai', content: '北京是一座很棒的城市！' },
    { type: 'human', content: '我的职业是软件工程师' },
    { type: 'ai', content: '软件工程师是一个很有前景的职业！' },
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

  // 按消息数量截断：只保留最近 maxMessages 条
  const trimmedMessages = allMessages.slice(-maxMessages);

  console.log(`截断后消息数量: ${trimmedMessages.length}`);
  console.log('保留的消息:');
  trimmedMessages.forEach((msg, i) => {
    const content =
      typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    console.log(`  ${i + 1}. [${msg.constructor.name}]: ${content}`);
  });
}

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

// =========== 2. 按 Token 数量截断 ===========
async function tokenCountTruncation() {
  console.log('\n========== 按 Token 数量截断 ==========\n');

  const history = new InMemoryChatMessageHistory();
  const maxTokens = 100; // 限制最多 100 个 token
  const enc = getEncoding('cl100k_base');

  // 模拟一段较长的对话
  const messages = [
    { type: 'human', content: '我叫李四' },
    { type: 'ai', content: '你好李四，很高兴认识你！' },
    { type: 'human', content: '我是一名设计师' },
    { type: 'ai', content: '设计师是个很有创造力的职业！你主要做什么类型的设计？' },
    { type: 'human', content: '我喜欢艺术和音乐' },
    { type: 'ai', content: '艺术和音乐都是很好的爱好，它们能激发创作灵感。' },
    { type: 'human', content: '我擅长 UI/UX 设计' },
    { type: 'ai', content: 'UI/UX 设计非常重要，好的用户体验能让产品更成功！' },
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
  console.log(`原始 token 数: ${countTokens(allMessages, enc)}`);

  // 使用 trimMessages API：使用 js-tiktoken 计算 token 数量
  const trimmedMessages = await trimMessages(allMessages, {
    maxTokens: maxTokens,
    tokenCounter: async (msgs) => countTokens(msgs, enc),
    strategy: 'last', // 保留最近的消息
  });

  const totalTokens = countTokens(trimmedMessages, enc);
  console.log(`总 token 数: ${totalTokens}/${maxTokens}`);
  console.log(`保留消息数量: ${trimmedMessages.length}`);
  console.log('保留的消息:');
  trimmedMessages.forEach((msg, i) => {
    const content =
      typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    const tokens = enc.encode(content).length;
    console.log(`  ${i + 1}. [${msg.constructor.name}] (${tokens} tokens): ${content}`);
  });
}

async function runAll() {
  await messageCountTruncation();
  await tokenCountTruncation();
}

runAll().catch(console.error);
