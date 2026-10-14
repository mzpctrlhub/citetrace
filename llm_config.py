# llm_config.py
"""
CiteTrace 统一大模型配置中心与调用客户端
- 支持标准 OpenAI 兼容协议 (Chat Completions)
- 支持自动模型别名映射 (如 ling-3.1-flash:free -> inclusionai/ling-3.1-flash:free)
- 支持 URL 智能归一化 (兼容包含 /chat/completions 的 Endpoint)
- 内置 429 限流与偶发超时的指数退避重试 (Exponential Backoff)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("citetrace.llm")

# 已知模型别名映射表
MODEL_ALIASES: Dict[str, str] = {
    "ling-3.1-flash:free": "inclusionai/ling-3.1-flash:free",
    "ling-3.1-flash": "inclusionai/ling-3.1-flash:free",
    "ling-3.0-flash:free": "inclusionai/ling-3.0-flash-sante:free",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
    "deepseek-chat": "deepseek-chat",
    "gpt-4o-mini": "gpt-4o-mini",
}


def normalize_base_url(url: str) -> str:
    """归一化 base_url，若传入完整 endpoint 则去除末尾 /chat/completions"""
    if not url:
        return "https://api.openai.com/v1"
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[:-len("/chat/completions")]
    return cleaned.rstrip("/")


def resolve_model_id(model_id: str) -> str:
    """解析模型名称（支持别名匹配）"""
    if not model_id:
        return "inclusionai/ling-3.1-flash:free"
    stripped = model_id.strip()
    return MODEL_ALIASES.get(stripped.lower(), stripped)


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """轻量读取 .env 文件（避免强依赖 python-dotenv）"""
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values
        
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("'\"")
    return values


@dataclass
class ModelEndpointConfig:
    provider: str = "openai_compatible"
    api_key: str = ""
    base_url: str = "https://api.commandcode.ai/provider/v1"
    model: str = "inclusionai/ling-3.1-flash:free"
    timeout: int = 60
    max_retries: int = 6


class LLMClient:
    """统一 OpenAI 兼容大模型客户端"""

    def __init__(self, config: ModelEndpointConfig):
        self.config = config
        self._openai_client = None
        self.last_raw_response: str = ""

    def _get_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self.config.api_key or "sk-placeholder",
                base_url=normalize_base_url(self.config.base_url),
                timeout=self.config.timeout,
            )
        return self._openai_client

    def chat_completion(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用大模型 Chat Completion
        内置 429 与网络异常的指数退避重试
        - max_tokens 默认为 None（不作人工硬性截断，交由模型原生上限与自然停止符决定）
        """
        if messages is None:
            messages = []
            
        target_model = resolve_model_id(model or self.config.model)
        client = self._get_client()

        last_error = None
        wait_seconds = 3.0

        for attempt in range(self.config.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if response_format:
                    kwargs["response_format"] = response_format


                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                content = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", "") or ""
                logger.info(
                    f"[LLM DEBUG] finish_reason={choice.finish_reason}, "
                    f"content_len={len(content)}, reasoning_len={len(reasoning)}"
                )
                self.last_raw_response = content.strip()
                return self.last_raw_response

            except Exception as e:
                last_error = e
                err_str = str(e)
                # 检查是否为限流或偶发网络错误
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                is_temp_unavailable = "503" in err_str or "temporarily unavailable" in err_str.lower()
                
                if (is_rate_limit or is_temp_unavailable) and attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"LLM 调用限流/排队 (尝试 {attempt+1}/{self.config.max_retries})，"
                        f"等待 {wait_seconds:.1f} 秒后重试... 错误: {e}"
                    )
                    time.sleep(wait_seconds)
                    wait_seconds *= 2.0
                else:
                    logger.error(f"LLM 调用失败: {e}")
                    if attempt >= self.config.max_retries - 1:
                        break
                    time.sleep(1.0)

        raise RuntimeError(f"LLM 调用经 {self.config.max_retries} 次重试后仍失败: {last_error}")

    def generate_json(
        self,
        *,
        prompt: str,
        payload: dict,
        schema,                 # type[BaseModel]
        system: str = "You are a precise academic citation parser. Output strictly valid JSON and nothing else.",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        repair_rounds: int = 1,
    ):
        """
        根据提示词模板与上下文渲染请求，并解析与校验目标 Pydantic Schema。
        内置 Markdown 剥壳、花括号定位提取与错误自愈修复轮。
        """
        rendered_prompt = render_prompt(prompt, payload)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": rendered_prompt},
        ]

        last_raw = ""
        last_err = None
        current_rounds = repair_rounds

        while True:
            raw_output = self.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = raw_output

            # 3. 剥壳: 去除代码围栏
            cleaned = raw_output.strip()
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if fence_match:
                cleaned = fence_match.group(1).strip()

            # 截取第一个 { 到最后一个 } 之间的子串
            start_brace = cleaned.find("{")
            end_brace = cleaned.rfind("}")
            if start_brace != -1 and end_brace != -1 and end_brace >= start_brace:
                cleaned = cleaned[start_brace : end_brace + 1]

            try:
                data = json.loads(cleaned)
                return schema.model_validate(data)
            except Exception as e:
                last_err = e
                logger.warning(f"JSON 解析/Schema校验失败: {e}。剩余修复轮次: {current_rounds}")
                if current_rounds > 0:
                    current_rounds -= 1
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({
                        "role": "user",
                        "content": f"你的上一次输出无法解析，错误是 {e}。请只输出修正后的完整 JSON。",
                    })
                else:
                    snippet = last_raw[:500] if len(last_raw) > 500 else last_raw
                    raise ValueError(f"JSON 解析经重试后仍失败: {last_err}。原始响应前500字:\n{snippet}") from last_err

def render_prompt(template: str, payload: Dict[str, Any]) -> str:
    """把 {{大写键名}} 占位符替换为 payload 中对应的值。"""
    res = template
    for k, v in payload.items():
        res = res.replace("{{" + k.upper() + "}}", str(v))
    return res


class ConfigHub:
    """全局配置注册中心与角色路由"""

    def __init__(self, env_path: Optional[Path] = None):
        self.env_vars = load_env_file(env_path)


    def get_role_config(self, role: str = "fast") -> Tuple[ModelEndpointConfig, str]:
        """
        根据角色获取模型配置
        role 可选: fast | default | reasoning
        """
        env = self.env_vars
        default_key = env.get("COMMANDCODE_API_KEY", os.getenv("COMMANDCODE_API_KEY", ""))
        default_url = env.get("COMMANDCODE_BASE_URL", os.getenv("COMMANDCODE_BASE_URL", "https://api.commandcode.ai/provider/v1"))

        if role == "fast" or role == "claim_extractor":
            # 轻量快速模型优先使用 Ling 3.1 Flash
            api_key = env.get("COMMANDCODE_FAST_API_KEY") or default_key
            base_url = env.get("COMMANDCODE_FAST_BASE_URL") or default_url
            raw_model = env.get("COMMANDCODE_FAST_MODEL") or "inclusionai/ling-3.1-flash:free"
            timeout = int(env.get("COMMANDCODE_FAST_TIMEOUT", "180"))
        elif role == "reasoning":
            api_key = env.get("COMMANDCODE_REASONING_API_KEY") or default_key
            base_url = env.get("COMMANDCODE_REASONING_BASE_URL") or default_url
            raw_model = env.get("COMMANDCODE_REASONING_MODEL") or "deepseek-chat"
            timeout = int(env.get("COMMANDCODE_REASONING_TIMEOUT", "90"))
        else:
            # 默认主力裁判模型
            api_key = default_key
            base_url = default_url
            raw_model = env.get("COMMANDCODE_MODEL") or "z-ai/glm-5.3-flash"
            timeout = int(env.get("COMMANDCODE_TIMEOUT", "60"))

        resolved_model = resolve_model_id(raw_model)
        cfg = ModelEndpointConfig(
            provider="openai_compatible",
            api_key=api_key,
            base_url=normalize_base_url(base_url),
            model=resolved_model,
            timeout=timeout,
        )
        return cfg, resolved_model

    def get_client(self, role: str = "fast") -> Tuple[LLMClient, str]:
        cfg, model_name = self.get_role_config(role)
        return LLMClient(cfg), model_name


# 全局单例辅助函数
_global_hub: Optional[ConfigHub] = None


def get_hub() -> ConfigHub:
    global _global_hub
    if _global_hub is None:
        _global_hub = ConfigHub()
    return _global_hub


def get_llm_client(role: str = "fast") -> Tuple[LLMClient, str]:
    """获取指定角色的 LLMClient 和解析后的模型名"""
    return get_hub().get_client(role)
