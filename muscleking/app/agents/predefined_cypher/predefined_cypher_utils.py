import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FitnessVectorQueryMatcher:
    """
    使用 TF-IDF 向量化实现的健身查询匹配器。

    用于将用户自然语言健身问题
    → 匹配到预定义的 Cypher 查询模板
    → 并抽取必要参数
    """

    def __init__(
        self,
        predefined_cypher_dict: Dict[str, str],
        query_descriptions: Dict[str, str],
        similarity_threshold: float = 0.5,
    ) -> None:
        self.predefined_cypher_dict = predefined_cypher_dict
        self.query_descriptions = query_descriptions
        self.similarity_threshold = similarity_threshold

        self._vectorizer = TfidfVectorizer()
        self._query_vectors = self._compute_query_vectors()

    # =========================
    # 向量化查询描述
    # =========================

    def _compute_query_vectors(self) -> Dict[str, np.ndarray]:
        keys: List[str] = []
        corpus: List[str] = []

        for query_name in self.predefined_cypher_dict:
            description = self.query_descriptions.get(query_name, "")
            keys.append(query_name)
            corpus.append(f"{query_name} {description}".strip())

        if not corpus:
            return {}

        matrix = self._vectorizer.fit_transform(corpus).toarray()
        return {
            key: np.asarray(vector, dtype=np.float32)
            for key, vector in zip(keys, matrix)
        }

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._vectorizer.get_feature_names_out())))
        return self._vectorizer.transform(texts).toarray()

    # =========================
    # 查询匹配
    # =========================

    def match_query(self, user_question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not user_question or not self._query_vectors:
            return []

        question_vector = self._embed([user_question])
        if question_vector.size == 0:
            return []

        question_vector = question_vector[0]

        similarities: List[tuple[str, float]] = []
        for query_name, vector in self._query_vectors.items():
            score = cosine_similarity([question_vector], [vector])[0][0]
            similarities.append((query_name, float(score)))

        similarities.sort(key=lambda item: item[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for query_name, score in similarities[:top_k]:
            if score >= self.similarity_threshold:
                results.append(
                    {
                        "query_name": query_name,
                        "similarity": score,
                        "cypher": self.predefined_cypher_dict[query_name],
                    }
                )
        return results

    # =========================
    # 参数抽取
    # =========================

    def extract_parameters(
        self,
        user_question: str,
        query_name: str,
        llm: Any | None = None,
    ) -> Dict[str, str]:
        """
        从用户问题中抽取 Cypher 所需参数
        """
        if query_name not in self.predefined_cypher_dict:
            return {}

        cypher_template = self.predefined_cypher_dict[query_name]
        param_names = re.findall(r"\$(\w+)", cypher_template)
        if not param_names:
            return {}

        # 优先使用 LLM
        if llm is not None:
            llm_params = self._extract_parameters_with_llm(
                user_question, param_names, query_name, llm
            )
            if llm_params:
                return llm_params

        # 兜底：规则抽取
        return self._extract_parameters_with_rules(user_question, param_names)

    # =========================
    # 规则参数抽取（健身版）
    # =========================

    @staticmethod
    def _extract_parameters_with_rules(
        user_question: str,
        param_names: List[str],
    ) -> Dict[str, str]:
        params: Dict[str, str] = {}

        for name in param_names:
            # 训练动作
            if name == "exercise_name":
                match = re.search(
                    r"(深蹲|卧推|硬拉|引体向上|俯卧撑|弯举|推举|划船)",
                    user_question,
                )
                if match:
                    params[name] = match.group(1)

            # 肌群
            elif name == "muscle_name":
                match = re.search(
                    r"(胸|背|腿|肩|手臂|肱二头|肱三头|核心|腹肌)",
                    user_question,
                )
                if match:
                    params[name] = match.group(1)

            # 器械
            elif name == "equipment_name":
                match = re.search(
                    r"(哑铃|杠铃|壶铃|拉力器|史密斯|徒手)",
                    user_question,
                )
                if match:
                    params[name] = match.group(1)

            # 难度
            elif name == "difficulty":
                match = re.search(r"(新手|初级|中级|高级|进阶)", user_question)
                if match:
                    params[name] = match.group(1)

        return params

    # =========================
    # LLM 参数抽取
    # =========================

    @staticmethod
    def _extract_parameters_with_llm(
        user_question: str,
        param_names: List[str],
        query_name: str,
        llm: Any,
    ) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是健身领域的参数提取助手，从用户问题中提取指定参数，只输出 JSON。",
                ),
                (
                    "human",
                    f"""用户问题: {user_question}
查询类型: {query_name}
需要提取的参数: {', '.join(param_names)}

请以 JSON 返回，例如:
{{"exercise_name": "深蹲", "muscle_name": "腿"}}""",
                ),
            ]
        )

        response = llm.invoke(prompt.format_prompt())
        content = getattr(response, "content", "") or ""

        try:
            match = re.search(r"{.*}", content, re.DOTALL)
            if not match:
                return {}
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return {
                    str(k): str(v)
                    for k, v in parsed.items()
                    if v is not None and str(v).strip()
                }
        except Exception as exc:  # pragma: no cover
            print(f"无法解析 LLM 参数 JSON: {exc}")

        return {}


# =========================
# 工厂方法
# =========================

def create_vector_query_matcher(
    predefined_cypher_dict: Dict[str, str],
    query_descriptions: Optional[Dict[str, str]] = None,
) -> FitnessVectorQueryMatcher:
    descriptions = query_descriptions or {
        key: key.replace("_", " ") for key in predefined_cypher_dict.keys()
    }
    return FitnessVectorQueryMatcher(predefined_cypher_dict, descriptions)
