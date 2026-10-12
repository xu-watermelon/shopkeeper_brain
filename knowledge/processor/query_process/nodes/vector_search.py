# knowledge/processor/query_process/nodes/vector_search_node.py

"""向量检索节点

对用户查询进行向量化，在 Milvus 中执行混合搜索（稠密 + 稀疏），返回相关切片。
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, item_name_filter


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_state(state)

        # 2. 获取嵌入模型
        try:
            embedding_molde = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state

        # 3. 获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return state

        # 4. 对问题嵌入
        try:
            embed_query = generate_bge_m3_hybrid_vectors(model=embedding_molde, embedding_documents=[validated_query])
        except Exception as e:
            self.logger.error(f"问题{validated_query}嵌入失败")
            return state

        # 5. 构建过滤表达式以及表达式参数
        filter_expr, filter_expr_param = item_name_filter(validate_item_names)

        try:
            # 6. 创建混合请求
            hybrid_search_request = create_hybrid_search_requests(dense_vector=embed_query['dense'][0],
 sparse_vector=embed_query['sparse'][0],expr=filter_expr,
 filter_expr_params=filter_expr_param)

            # 7. 执行混合搜索请求
            hybrid_search_reps = execute_hybrid_search_query(milvus_client=milvus_client,
collection_name=self.config.chunks_collection,
search_requests=hybrid_search_request,output_fields=['chunk_id', 'content', 'item_name','title'])

            # 8. 获取搜索结果
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state

            # 9. 更新state
            state['embedding_chunks'] = hybrid_search_reps[0]

            # 10. 返回
            return state
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return state

    def _validate_state(self, state: QueryGraphState)->Tuple[str,List[str]]:
        """
        校验输入参数
        Args:
            state:

        Returns:

        """
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        # 3. 返回
        return rewritten_query, item_names


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks',[]):
        print(json.dumps(r, ensure_ascii=False, indent=2))