# knowledge/processor/query_process/nodes/rerank_node.py

"""Rerank 重排序节点

使用 Reranker 模型对 RRF 融合结果和网络搜索结果进行重排序，
并通过优化版断崖检测实现动态 TopK 截断。
"""

from typing import Dict, Any, List, Tuple
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, setup_logging, T
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    """Rerank 重排序节点

    流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
    """
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行重排序"""
        # 1. 获取 query
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 2. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_source_docs(state)

        # 3. Rerank 精排
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)

        # 4. 动态 Top_K 截取(断崖检测 + 绝对分数底线过滤)
        cutoff_docs = self._cliff_cutoff(reranked_docs)

        state['reranked_docs'] = cutoff_docs
        return state

    def _cliff_catoff(self, reranked_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """断崖检测动态截断
        扫描所有相邻文档分数差，找到最大落差位置进行截断，
        同时保证至少返回 lower_bound 个文档。

        参数:
            reranked_docs: 按分数降序排列的文档列表
            rerank_min_top_k: 最少返回文档数
            rerank_max_top_k: 最多返回文档数

        返回值:
            截断后的文档列表
        """
        upper_bound = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)

        if upper_bound <= 1:
            return reranked_docs[:upper_bound]

        cut_off = upper_bound
        max_gap = 0.0

        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 分数差值
            gap = current_score - next_score

            if gap >= self.config.rerank_gap_abs and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
                self.logger.info(f"位置{cut_off}发生断崖，gap={gap:.4f}")
        # 兜底：不管断崖在哪，至少保留lower_bound个
        cut_off = max(cut_off, lower_bound)
        return reranked_docs[:cut_off] #左闭开区间：包含起始位置，不包含结束位置
    
    def _merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 文档和 Web 搜索文档"""
        final_docs = []

        for rrf_doc in (state.get('rrf_chunks') or []):
            if not isinstance(rrf_doc, dict):
                continue
            content = rrf_doc.get('content', '').strip()
            if not content:
                continue
            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id')
            format_rrf_doc = self._format_rrf_docs(
                content=content, title=title, chunk_id=chunk_id, source="local"
            )
            final_docs.append(format_rrf_doc)

        for web_doc in (state.get('web_search_docs') or []):
            if not isinstance(web_doc, dict):
                continue
            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()
            if not content:
                continue
            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()
            format_web_doc = self._format_rrf_docs(
                content=content, title=title, url=url, source="web"
            )
            final_docs.append(format_web_doc)

        self.logger.info(f"收集到准备进行 Rerank 精排的文档 {len(final_docs)}")
        return final_docs

    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None,
                        url: str = "", source: str = "") -> Dict[str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }

    @staticmethod #避免实例方法调用时第一个参数传入self
    def _sigmoid(score: float) -> float:
        """归一化，将( -∞ , +∞ )   映射到 （0,1）
            exp(x) 就是以自然常数 e=2.71828 为底的指数函数
            exp(10) ≈ 22026  →  1 / (1 + 22026) ≈ 0.0000454  ≈ 0
            exp(-10) ≈ 0.0000454  →  1 / (1 + 0.0000454) ≈ 0.99995  ≈ 1
        """
        return 1.0 / (1.0 + math.exp(-score))
    
    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""
        if not merged_multi_docs:
            return []

        rerank_client = AIClients.get_bge_m3_rerank_client()
        if rerank_client is None:
            self.logger.error("重排序模型获取失败")
            return []

        query_doc_content_pairs = [(user_query, doc.get('content')) for doc in merged_multi_docs]

        try:
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_content_pairs)

            # 单文档防护
            if isinstance(rerank_scores, (float, int)):
                rerank_scores = [rerank_scores]
            print(rerank_scores)  # [5.06640625, -4.7421875, 3.962890625, -9.4765625]

            # 没做归一化处理,用于测试比较
            score_docs1 = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            print(f"没做归一化处理:{score_docs1}")

            # 归一化处理
            score_docs = [{**doc, "score": self._sigmoid(score)} for doc, score in
                          zip(merged_multi_docs, rerank_scores)]
            print(f"归一化处理:{score_docs}")

            sorted_score_docs = sorted(score_docs, key=lambda x: x["score"], reverse=True)
            print(f"排序后结果返回:{sorted_score_docs}")
            return sorted_score_docs

        except Exception as e:
            self.logger.error(f"Rerank 重排序失败: {str(e)}")
            return [{**doc, "score": None} for doc in merged_multi_docs]


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")