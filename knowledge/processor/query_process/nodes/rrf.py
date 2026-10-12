# knowledge/processor/query_process/nodes/rrf_node.py

"""RRF 融合排序节点

使用 Reciprocal Rank Fusion 算法融合多路检索结果。
"""

from typing import Dict, Any, List, Tuple

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, setup_logging


class RrfNode(BaseNode):
    """RRF 融合排序节点

    流程: 收集多路检索结果 → 带权重 RRF 融合 → 按得分降序返回
    """
    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 RRF 融合

        Args:
            state: 需包含 embedding_chunks 和 hyde_embedding_chunks

        Returns:
            更新后的 state，包含 rrf_chunks
        """
        # 1. 各路搜索的结果
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get('embedding_chunks') or []
        # 1.2 获取 hyde 向量检索路的结果
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        # 2. 为不同路的搜索结果设置不同的权重
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
        }

        # 3. 构建 rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4. 利用 RRF 的计算公式去获取到所有路查询到的所有 chunk 对应的 score
        rrf_merge_results: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(
            rrf_inputs, self._rrf_k, self._top_k
        )

        # 5. 获取 rrf_chunks（只取文档，不要分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新 state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回 state
        return state

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一处理各路检索到的结果

        Args:
            rrf_input: 各路不同数据结构的检索结果

        Returns:
            统一处理后的标准数据结构的检索结果
        """
        diff_path_result = []

        # 1. 遍历各路搜索结果
        if not rrf_input:
            return []

        # 2. 遍历该路的所有结果
        for doc in rrf_input:
            # 2.1 判断是否有效
            if not isinstance(doc, dict):
                continue

            # 2.2 获取 entity
            entity = doc.get('entity')
            if not entity:
                continue

            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(self, rrf_inputs, _rrf_k, _top_k) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用 RRF 公式计算每一个文档的总得分

        Args:
            rrf_inputs: 各路的搜索结果 + 权重
            _rrf_k: 平滑参数，通常取 60
            _top_k: 合并完之后返回的文档数

        Returns:
            合并以及排序后的文档列表
        """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}    # 存放所有 chunk 的文档数据

        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue

                # RRF 公式: score += weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (_rrf_k + i)

                # 使用 setdefault 保留首次遇到的文档版本
                chunk_data.setdefault(chunk_id, doc)

        # 按得分降序排序动态截取前 top_k 条
        sorted_results = sorted(
            [(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()],
            key=lambda x: x[1],#根据score进行排序
            reverse=True
        )

        return sorted_results[:_top_k] if _top_k else sorted_results


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")