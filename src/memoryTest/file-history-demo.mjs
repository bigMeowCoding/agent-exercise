import { ChatOpenAI } from '@langchain/openai';
import { FileSystemChatMessageHistory } from '@langchain/community/stores/message/file_system';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import path from 'node:path';

const model = new ChatOpenAI({
  modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
  apiKey: process.env.MOONSHOT_API_KEY,
  temperature: 1,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL ?? 'https://api.moonshot.cn/v1',
  },
});

async function fileHistoryDemo() {
  // 文件路径：当前工作目录下的 chat_history.json
  const filePath = path.join(process.cwd(), 'chat_history.json');
  const sessionId = 'user_session_001';

  // 系统提示
  const systemMessage = new SystemMessage(
    '你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。',
  );

  // ========== 第一轮对话 ==========
  console.log('[第一轮对话]');
  const history = new FileSystemChatMessageHistory({
    filePath,
    sessionId,
  });

  const userMessage1 = new HumanMessage('红烧肉怎么做？');
  await history.addMessage(userMessage1);

  const messages1 = [systemMessage, ...(await history.getMessages())];
  const response1 = await model.invoke(messages1);
  await history.addMessage(response1);

  console.log(`用户: ${userMessage1.content}`);
  console.log(`助手: ${response1.content}`);
  console.log(`✓ 对话已保存到文件: ${filePath}\n`);

  // ========== 第二轮对话 ==========
  console.log('[第二轮对话]');
  const userMessage2 = new HumanMessage('好吃吗？');
  await history.addMessage(userMessage2);

  const messages2 = [systemMessage, ...(await history.getMessages())];
  const response2 = await model.invoke(messages2);
  await history.addMessage(response2);

  console.log(`用户: ${userMessage2.content}`);
  console.log(`助手: ${response2.content}`);
  console.log('✓ 对话已更新到文件\n');

  // ========== 从文件恢复历史并继续对话 ==========
  console.log('[从文件恢复历史]');
  const restoredHistory = new FileSystemChatMessageHistory({
    filePath,
    sessionId,
  });
  const restoredMessages = await restoredHistory.getMessages();
  console.log(`从文件恢复了 ${restoredMessages.length} 条历史消息：`);
  restoredMessages.forEach((msg, index) => {
    const prefix = msg.type === 'human' ? '用户' : '助手';
    const content =
      typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    console.log(`  ${index + 1}. [${prefix}]: ${content.substring(0, 50)}...`);
  });
  console.log('');

  // ========== 第三轮对话 ==========
  console.log('[第三轮对话]');
  const userMessage3 = new HumanMessage('需要哪些食材？');
  await restoredHistory.addMessage(userMessage3);

  const messages3 = [systemMessage, ...(await restoredHistory.getMessages())];
  const response3 = await model.invoke(messages3);
  await restoredHistory.addMessage(response3);

  console.log(`用户: ${userMessage3.content}`);
  console.log(`助手: ${response3.content}`);
  console.log('✓ 对话已保存到文件\n');
}

fileHistoryDemo().catch(console.error);
