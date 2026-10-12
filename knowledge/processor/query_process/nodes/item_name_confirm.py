# knowledge/processor/query_process/nodes/item_name_confirm_node.py

"""商品名称确认节点

从用户查询中提取商品名称，通过向量相似度匹配与数据库中已有商品对齐确认。
"""

import logging
import json
import re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query
)
from knowledge.utils.embedding_util import (
    generate_bge_m3_hybrid_vectors
)
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.mongo_history_util import (
    get_recent_messages,
    update_message_item_names
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItemNameAligner:
    """
    商品名对齐器
    主要职责：
    1. 查询向量数据库
    2. 评分对齐
    3. 分数差异过滤
    """

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """执行匹配、对齐、过滤三步流程"""
        # 1. 查询向量数据库
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_result)

        # 3. 分数差异过滤（仅当 confirmed 有多个时）
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """根据 LLM 提取的商品名，查询向量数据库"""
        search_results = []

        milvus_client = StorageClients.get_milvus_client()
        if milvus_client is None:
            return search_results

        embedding_model = AIClients.get_bge_m3_client()
        if embedding_model is None:
            logger.error(f"获取嵌入模型失败")
            return search_results

        hybrid_embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)

        for index, extract_item_name in enumerate(item_names):
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result['dense'][index],
                sparse_vector=hybrid_embedding_result['sparse'][index],
            )

            hybrid_search_result = execute_hybrid_search_query(
                milvus_client,
                collection_name="kb_item_names_v2",
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"]
            )

            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            search_results.append(item_name_search_result)

        return search_results

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """根据评分对齐商品名称"""
        confirmed = []
        options = []

        for item_name_search_result in search_results:
            extracted_name = item_name_search_result.get('extracted_name')
            matches = sorted(item_name_search_result.get('matches'), key=lambda x: x['score'], reverse=True)
            #这段代码是商品名称评分对齐逻辑，根据向量搜索的分数决定商品名的确认或候选：
            #高分区（≥0.7）：
            #  若存在与提取名称完全匹配的 → 直接确认
            #  若仅1条高分结果 → 直接确认
            #  若多条高分 → 取前3条作为候选选项
            #中分区（0.6~0.7）：无高分时，取中分结果前3条作为候选选项
            #返回：confirmed（确认列表）和 options（候选列表，最多3个
            high = [m for m in matches if m.get('score') >= 0.7]

            if high:
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)

                if extract:
                    picked = extract["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    for h in high[:3]:
                        picked = h.get('item_name')
                        if picked not in options and picked not in confirmed:
                            options.append(picked)
            else:
                mid = [m for m in matches
                       if m['score'] >= 0.6 and m.get('item_name') not in options and m.get('item_name') not in confirmed]
                if mid:
                    for m in mid[:3]:
                        picked = m.get('item_name')
                        options.append(picked)

        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]):
        """分数差异过滤，剔除误判"""
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        max_item_name_score = sorted_item_name_score[0][1]

        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]


class ItemNameExtractor:
    """
    商品名提取器
    基于用户的原始问题和历史对话提取用户真正想问的商品名
    """

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """LLM 根据用户原始问题提取商品名"""
        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        llm_client = AIClients.get_llm_client(response_format=True)
        if llm_client is None:
            return result

        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text if history_text else "暂无上下文",
            query=original_query
        )
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        llm_content = llm_response.content.strip()

        if not llm_content.strip():
            return result

        try:
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")
        except Exception as e:
            logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        return result

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            raw_item_names = parsed_llm_result.get('item_names')
            if not isinstance(raw_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in raw_item_names if raw_item.strip()]

            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点

    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 历史回填
    """
    name = "item_name_confirm_node"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state.get("original_query")
        session_id = state.get('session_id')

        # 2. 获取历史对话
        chat_history = get_recent_messages(session_id, limit=10)
        history_text = ""
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("text", "")
            history_text += f"{role}: {content}\n"

        # 3. 调用 LLM 提取商品名
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        item_names = clean_llm_result.get('item_names')
        rewritten_query = clean_llm_result.get('rewritten_query')

        # 4. 查询向量数据库 && 过滤
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 5. 决策分支，更新 state
        self._decide(state, item_names, confirmed, options, rewritten_query)

        # 6. 历史回填
        if confirmed:
            ids_to_update = [
                str(msg["_id"]) for msg in chat_history if not msg.get("item_names")
            ]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:
                    self.logger.warning(f"回填历史 item_names 失败: {e}")

        # 将历史对话写入 state，供下游使用
        state["history"] = chat_history

        return state

    def _decide(self, state: QueryGraphState, item_names: List[str],
                confirmed: List[str], options: List[str], rewritten_query: str):
        """根据对齐结果更新 state"""
        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
        elif options:
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    test_state: QueryGraphState = {
        "original_query": "RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？"
    }
    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)
    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")