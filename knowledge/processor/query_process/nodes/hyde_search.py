# knowledge/processor/query_process/nodes/hyde_search_node.py

"""HyDE 检索节点

使用 Hypothetical Document Embedding 技术：
先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量。
"""

import json
import logging
from typing import List, Tuple, Union, Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import (
    get_milvus_client,
    create_hybrid_search_requests,
    execute_hybrid_search_query
)
from knowledge.utils.bge_m3_embedding_util import (
    generate_hybrid_embeddings,
    get_bge_m3_embedding_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyDeSearchNode(BaseNode):
    """HyDE 检索节点

    流程: 参数校验 → LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        """执行 HyDE 检索

        Args:
            state: 需包含 rewritten_query 和 item_names

        Returns:
            {"hyde_embedding_chunks": [...]} 搜索结果列表
        """
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # 3. 获取嵌入模型 & milvus 客户端
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()
        if not embedding_model or not milvus_client:
            return state

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_hybrid_embeddings(
            embedding_model,
            embedding_documents=[embedding_document]
        )

        if not embedding_result:
            return state

        # 5. 获取 item_name 的过滤表达式
        item_name_filtered_expr = self._item_name_filte_expr(validate_item_names)

        # 6. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filtered_expr
        )

        # 7. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id", "content", "item_name",'title']
        )

        if not reps or not reps[0]:
            return state

        # 8. 只更新 hyde_embedding_chunks
        return {"hyde_embedding_chunks": reps[0]}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        return rewritten_query, item_names

  

    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        """构建商品名过滤表达式"""
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        return f" item_name in [{quoted}]"


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")