import fs from 'node:fs/promises';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { z } from 'zod';

const model = new ChatOpenAI({
    modelName: process.env.KIMI_MODEL ?? 'kimi-k2.6',
    apiKey: process.env.MOONSHOT_API_KEY,
    configuration: {
        baseURL: 'https://api.moonshot.cn/v1',
    },
});

const readFileTool = tool(
    async ({ filePath }) => {
        const content = await fs.readFile(filePath, 'utf-8');
        console.log(`  [工具调用] read_file("${filePath}") - 成功读取 ${content.length} 字节`);
        return `文件内容:\n${content}`;
    },
    {
        name: 'read_file',
        description: '用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。输入文件路径（可以是相对路径或绝对路径）。',
        schema: z.object({
            filePath: z.string().describe('要读取的文件路径'),
        }),
    }
);

const tools = [readFileTool];
const toolsByName = Object.fromEntries(tools.map((t) => [t.name, t]));
const modelWithTools = model.bindTools(tools);

const messages = [
    new SystemMessage('你是一个代码助手，可以使用工具读取文件并解释代码。'),
    new HumanMessage('请读取 src/tool-file-read.mjs 文件，并解释这段代码做了什么'),
];

let response = await modelWithTools.invoke(messages);
messages.push(response);

while (response.tool_calls?.length) {
    for (const toolCall of response.tool_calls) {
        const selectedTool = toolsByName[toolCall.name];
        const result = selectedTool
            ? await selectedTool.invoke(toolCall.args)
            : `未知工具: ${toolCall.name}`;
        messages.push(new ToolMessage({ content: result, tool_call_id: toolCall.id }));
    }
    response = await modelWithTools.invoke(messages);
    messages.push(response);
}

console.log(response.content);
