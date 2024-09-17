import bentoml
import json

from openai_endpoints import openai_api_app

from openai import OpenAI


MAX_TOKENS = 1024
SYSTEM_PROMPT = "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses only based on the results from these function calls. The results will be specified in the 'converted_amount' field. The source and target currencies will be specified in  the 'from_currency' and 'to_currency' respectively."
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"


@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class Llama:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from lmdeploy.serve.async_engine import AsyncEngine
        from lmdeploy.messages import TurbomindEngineConfig

        engine_config = TurbomindEngineConfig(
            model_name=MODEL_ID,
            model_format="awq",
            cache_max_entry_count=0.85,
            enable_prefix_caching=True,
        )
        self.engine = AsyncEngine(MODEL_ID, backend_config=engine_config)

        import lmdeploy.serve.openai.api_server as lmdeploy_api_sever
        lmdeploy_api_sever.VariableInterface.async_engine = self.engine

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.stop_tokens = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.eos_token_id,
            ),
            "<|eot_id|>",
        ]


@bentoml.service(resources={"cpu": "1"})
class ExchangeAssistant:
    llm = bentoml.depends(Llama)
    
    def __init__(self):
        self.client = OpenAI(
            base_url=f"{self.llm.client_url}/v1",
            http_client=self.llm.to_sync.client,
            api_key="API_TOKEN_NOT_NEEDED"
        )

    def convert_currency(self, from_currency: str = "USD", to_currency: str = "CAD", amount: float = 1) -> float:
        exchange_rate = 3.14159 # Replace with actual exchange rate API
        return json.dumps({"from_currency": from_currency, "to_currency": to_currency, "converted_amount": round(float(amount) * exchange_rate, 2)})
    
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
            return response_message.content
