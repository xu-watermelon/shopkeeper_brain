"""MCP 网络搜索节点

通过 MCP 协议调用网络搜索服务获取外部信息。
"""

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from agents.mcp import MCPServerSse
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    """MCP 网络搜索节点。

    负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
    MCP 形式调用第三方的各种通用的搜索工具。
    """

    name = "mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """执行网络搜索。

        Args:
            state: 查询图状态，需包含 rewritten_query 和 item_names。

        Returns:
            更新后的状态，包含 web_search_docs 搜索结果。
        """
        # 1. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 2. 创建mcp_client 并且让客户端执行工具 bailian_web_search
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3. 只更新 state 的 web_search_docs
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数。

        Args:
            state: 查询图状态。

        Returns:
            校验后的 rewritten_query 和 item_names。

        Raises:
            StateFieldError: 参数校验失败时抛出。
        """
        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取 state 的 item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """异步执行 MCP 网络搜索。

        Args:
            validated_rewritten_query: 校验后的查询文本。

        Returns:
            搜索结果列表，每项包含 title、url、snippet。
        """
        # 1. 创建 MCP 客户端（SSE 模式）
        mcp_client = MCPServerSse(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,  # 服务端的端点
                "headers": {"Authorization": self.config.openai_api_key}  # 认证权限 api_key
            },
            cache_tools_list=True,  # 缓存工具列表
        )

        try:
            # 2. 建立 MCP 连接
            await mcp_client.connect()

            # 3. 执行工具
            execute_tool_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_rewritten_query, "count": 3}
            )

            # 4. 解析工具执行完的结果
            if not execute_tool_result:
                return []
            if not execute_tool_result.content[0]:
                return []

            # 4.1 获取 TextContent 对象的 text
            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                return []

            # 4.2 反序列化
            try:
                text_content_text: Dict[str, Any] = json.loads(text_content_text)

                # a) 获取 pages
                pages = text_content_text.get('pages', "")
                if not pages:
                    return []

                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})

                # c) 最终返回
                return search_result

            except Exception as e:
                self.logger.error("反序列MCP结果失败")
                return []

        finally:
            await mcp_client.cleanup()  # 关闭连接