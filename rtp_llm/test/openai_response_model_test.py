import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Optional


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "openai" / "openai_endpoint.py"
)


class _Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _CompleteResponseAsyncGenerator:
    def __init__(self, generator, collect_complete_response_func):
        self._generator = generator
        self._collect_complete_response_func = collect_complete_response_func
        self._all_responses = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        response = await self._generator.__anext__()
        self._all_responses.append(response)
        return response

    async def gen_complete_response_once(self):
        async def generate_from_list():
            for response in self._all_responses:
                yield response

        return await self._collect_complete_response_func(generate_from_list())


def _load_endpoint_class():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    endpoint_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenaiEndpoint"
    )
    method_names = {
        "_collect_complete_response",
        "_complete_stream_response",
        "chat_completion",
        "_render_single_output",
    }
    method_nodes = [
        node
        for node in endpoint_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    if {node.name for node in method_nodes} != method_names:
        raise AssertionError("OpenaiEndpoint response methods are missing")

    class_node = ast.ClassDef(
        name="OpenaiEndpoint",
        bases=[ast.Name(id="object", ctx=ast.Load())],
        keywords=[],
        body=method_nodes,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[class_node], type_ignores=[]))
    namespace = {
        "Any": Any,
        "AsyncGenerator": AsyncGenerator,
        "ChatCompletionRequest": Any,
        "ChatCompletionResponse": _Record,
        "ChatCompletionResponseChoice": _Record,
        "ChatCompletionStreamResponse": _Record,
        "ChatMessage": _Record,
        "CompleteResponseAsyncGenerator": _CompleteResponseAsyncGenerator,
        "DebugInfo": _Record,
        "ExceptionType": SimpleNamespace(ERROR_INPUT_FORMAT_ERROR=1),
        "FtRuntimeException": RuntimeError,
        "Optional": Optional,
        "Request": Any,
        "RoleEnum": SimpleNamespace(assistant="assistant"),
        "StreamResponseObject": Any,
        "UsageInfo": _Record,
        "extract_request_headers": lambda headers: headers,
        "logging": logging,
        "parse_and_fill_banned_combo": lambda *args: None,
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["OpenaiEndpoint"]


OpenaiEndpoint = _load_endpoint_class()


def _stream_object(prompt_logits=None):
    choice = _Record(
        delta=_Record(
            role="assistant",
            content="answer",
            reasoning_content=None,
            function_call=None,
            tool_calls=None,
        ),
        finish_reason="stop",
        logprobs=None,
    )
    return _Record(
        choices=[choice],
        usage=_Record(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        aux_info=None,
        extra_outputs=None,
        prompt_logits=prompt_logits,
    )


async def _response_stream(prompt_logits=None, count=1):
    for _ in range(count):
        yield _stream_object(prompt_logits)


class OpenaiResponseModelTest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_model_reaches_stream_and_aggregate(self):
        prompt_logits = {"tokens": [7], "logits": [0.25]}
        responses = OpenaiEndpoint._complete_stream_response(
            _response_stream(count=2),
            None,
            model_name="request-model",
        )

        chunks = [chunk async for chunk in responses]
        self.assertEqual(
            ["request-model", "request-model"],
            [chunk.model for chunk in chunks],
        )

        aggregate_responses = OpenaiEndpoint._complete_stream_response(
            _response_stream(prompt_logits),
            None,
            model_name="request-model",
        )
        _ = [chunk async for chunk in aggregate_responses]
        aggregate = await aggregate_responses.gen_complete_response_once()
        self.assertEqual("request-model", aggregate.model)
        self.assertEqual(prompt_logits, aggregate.prompt_logprobs)

    async def test_default_model_parameter_remains_backward_compatible(self):
        responses = OpenaiEndpoint._complete_stream_response(
            _response_stream(), None
        )

        chunks = [chunk async for chunk in responses]
        self.assertEqual([""], [getattr(chunk, "model", None) for chunk in chunks])
        aggregate = await responses.gen_complete_response_once()
        self.assertEqual("", aggregate.model)

    def test_chat_completion_selects_request_model_then_configured_model(self):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        renderer = _Record(generate_choice=lambda *args, **kwargs: "choices")
        endpoint.chat_renderer = renderer
        endpoint.template_renderer = renderer
        endpoint.model_name = "configured-model"
        endpoint.tokenizer = _Record(encode=lambda value: [1])
        endpoint.backend_rpc_server_visitor = object()
        endpoint.render_chat = lambda request: _Record(
            rendered_prompt="", input_ids=[1], multimodal_inputs=[]
        )
        endpoint._extract_generation_config = lambda request: _Record(
            return_prompt_logits=False, sp_advice_prompt=""
        )
        endpoint._apply_renderer_chat_constraints = lambda *args: None
        endpoint._complete_stream_response = (
            lambda generator, debug_info, tokenizer, model_name="": model_name
        )
        raw_request = _Record(headers={})

        for requested_model, expected_model in (
            ("request-model", "request-model"),
            (None, "configured-model"),
        ):
            with self.subTest(requested_model=requested_model):
                request = _Record(
                    user_template=False,
                    model=requested_model,
                    debug_info=False,
                )
                self.assertEqual(
                    expected_model,
                    endpoint.chat_completion(1, request, raw_request),
                )

    async def test_batch_response_selects_request_model_then_configured_model(self):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)

        async def merge_outputs(output_generator):
            return output_generator

        renderer = _Record(
            _merge_non_streaming_outputs=merge_outputs,
            render_response_stream=lambda *args: "choices",
        )
        endpoint.chat_renderer = renderer
        endpoint.template_renderer = renderer
        endpoint.model_name = "configured-model"
        endpoint.tokenizer = object()

        async def collect_response(
            generator, debug_info, tokenizer, model_name=""
        ):
            return _Record(model=model_name)

        endpoint._collect_complete_response = collect_response
        generate_config = _Record(return_prompt_logits=False)

        for requested_model, expected_model in (
            ("request-model", "request-model"),
            (None, "configured-model"),
        ):
            with self.subTest(requested_model=requested_model):
                request = _Record(user_template=False, model=requested_model)
                response = await endpoint._render_single_output(
                    object(), request, generate_config
                )
                self.assertEqual(expected_model, response.model)


if __name__ == "__main__":
    unittest.main()
