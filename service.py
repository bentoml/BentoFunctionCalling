import uuid
import json
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints
from openai import OpenAI


MAX_TOKENS = 1024
SYSTEM_PROMPT = "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls."
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

@openai_endpoints(
    model_id=MODEL_ID,
    default_chat_completion_parameters=dict(stop=["<|eot_id|>"]),
)
@bentoml.service(
    name="bentovllm-llama3.1-70b-insruct-awq-service",
    traffic={
        "timeout": 1200,
        "concurrency": 256,  # Matches the default max_num_seqs in the VLLM engine
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class Llama:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_TOKENS,
            quantization="AWQ",
            enable_prefix_caching=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=self.stop_token_ids,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)


@bentoml.service(resources={"cpu": "1"})
class ExchangeAssistant:
    llm = bentoml.depends(Llama)
    
    def __init__(self):
        self.client = OpenAI(base_url=f"{Llama.url}/v1", api_key="API_TOKEN_NOT_NEEDED")

    def convert_currency(self, from_currency: str = "USD", to_currency: str = "CAD", amount: float = 1) -> float:
        exchange_rate = 3.14159 # Replace with actual exchange rate API
        return json.dumps({"converted_amount": round(float(amount) * exchange_rate, 2)})
    
    @bentoml.api
    def exchange(self, query: str = "I want to exchange 42 US dollars to Canadian dollars") -> str:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "convert_currency",
                    "description": "Convert from one currency to another. Result is returned in the 'converted_amount' key.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_currency": {"type": "string", "description": "The source currency to convert from, e.g. USD",},
                            "to_currency": {"type": "string", "description": "The target currency to convert to, e.g. CAD",},
                            "amount": {"type": "number", "description": "The amount to be converted"},
                        },
                        "required": [],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response_message = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=tools,
        ).choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "convert_currency": self.convert_currency,
            }
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    from_currency=function_args.get("from_currency"),
                    to_currency=function_args.get("to_currency"),
                    amount=function_args.get("amount"),
                )
                messages.append(
                    {
                        "role": "user",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            final_response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
            )
            return final_response.choices[0].message.content
        else:
            return "Unable to use the available tools."
