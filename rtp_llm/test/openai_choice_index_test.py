import ast
import copy
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, Optional

from pydantic import BaseModel, Field, ValidationError


RTP_LLM_ROOT = Path(__file__).resolve().parents[1]
API_DATATYPE_SOURCE = RTP_LLM_ROOT / "openai" / "api_datatype.py"
ENDPOINT_SOURCE = RTP_LLM_ROOT / "openai" / "openai_endpoint.py"


class _Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Message(_Record):
    def __init__(
        self,
        role=None,
        content=None,
        reasoning_content=None,
        function_call=None,
        tool_calls=None,
    ):
        super().__init__(
            role=role,
            content=content,
            reasoning_content=reasoning_content,
            function_call=function_call,
            tool_calls=tool_calls,
        )


class _AggregatedChoice(_Record):
    def __init__(self, index, message, finish_reason=None, logprobs=None):
        super().__init__(
            index=index,
            message=message,
            finish_reason=finish_reason,
            logprobs=logprobs,
        )


class _Logprobs(_Record):
    def __init__(self, *tokens):
        super().__init__(content=list(tokens), refusal=None)

    def model_copy(self, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)


def _find_class(source_path, class_name):
    tree = ast.parse(
        source_path.read_text(encoding="utf-8"), filename=str(source_path)
    )
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _load_stream_choice_class():
    class_node = _find_class(
        API_DATATYPE_SOURCE, "ChatCompletionResponseStreamChoice"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[class_node], type_ignores=[])
    )
    namespace = {
        "__name__": "openai_choice_schema_under_test",
        "BaseModel": BaseModel,
        "ChoiceLogprobs": Any,
        "DeltaMessage": Any,
        "Field": Field,
        "FinisheReason": Any,
        "Optional": Optional,
    }
    exec(compile(module, str(API_DATATYPE_SOURCE), "exec"), namespace)
    return namespace["ChatCompletionResponseStreamChoice"]


def _load_collector_class():
    endpoint_node = _find_class(ENDPOINT_SOURCE, "OpenaiEndpoint")
    collector_node = next(
        node
        for node in endpoint_node.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_collect_complete_response"
    )
    class_node = ast.ClassDef(
        name="OpenaiEndpoint",
        bases=[ast.Name(id="object", ctx=ast.Load())],
        keywords=[],
        body=[collector_node],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[class_node], type_ignores=[])
    )
    namespace = {
        "Any": Any,
        "AsyncGenerator": AsyncGenerator,
        "ChatCompletionResponse": _Record,
        "ChatCompletionResponseChoice": _AggregatedChoice,
        "ChatMessage": _Message,
        "DebugInfo": Any,
        "Dict": Dict,
        "Optional": Optional,
        "RoleEnum": SimpleNamespace(assistant="assistant"),
        "StreamResponseObject": Any,
        "UsageInfo": _Record,
        "logging": logging,
    }
    exec(compile(module, str(ENDPOINT_SOURCE), "exec"), namespace)
    endpoint_class = namespace["OpenaiEndpoint"]
    endpoint_class._merge_tool_calls = staticmethod(
        lambda existing_tool_calls, delta_tool_calls: existing_tool_calls
    )
    return endpoint_class


StreamChoice = _load_stream_choice_class()
OpenaiEndpoint = _load_collector_class()


def _choice(
    index,
    *,
    content=None,
    reasoning_content=None,
    finish_reason=None,
    logprobs=None,
):
    return _Record(
        index=index,
        delta=_Record(
            role=None,
            content=content,
            reasoning_content=reasoning_content,
            function_call=None,
            tool_calls=None,
        ),
        finish_reason=finish_reason,
        logprobs=logprobs,
    )


def _response(*choices):
    return _Record(
        choices=list(choices),
        usage=_Record(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        aux_info=None,
        extra_outputs=None,
    )


async def _responses(*responses):
    for response in responses:
        yield response


class OpenaiChoiceIndexTest(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, *responses):
        return await OpenaiEndpoint._collect_complete_response(
            _responses(*responses),
            debug_info=None,
            model_name="test-model",
        )

    def test_stream_choice_index_is_strict_non_negative_integer(self):
        for invalid_index in (-1, True, 1.0, "3"):
            with self.subTest(index=invalid_index):
                with self.assertRaises(ValidationError):
                    StreamChoice(index=invalid_index, delta=object())

        valid = StreamChoice(index=3, delta=object())
        self.assertEqual(3, valid.index)

    async def test_reordered_choices_merge_by_index_and_sort(self):
        result = await self._collect(
            _response(
                _choice(1, content="B1", logprobs=_Logprobs("b1")),
                _choice(0, content="A1", logprobs=_Logprobs("a1")),
            ),
            _response(
                _choice(
                    0,
                    content="A2",
                    finish_reason="stop",
                    logprobs=_Logprobs("a2"),
                ),
                _choice(
                    1,
                    content="B2",
                    finish_reason="length",
                    logprobs=_Logprobs("b2"),
                ),
            ),
        )

        self.assertEqual([0, 1], [choice.index for choice in result.choices])
        self.assertEqual(
            ["A1A2", "B1B2"],
            [choice.message.content for choice in result.choices],
        )
        self.assertEqual(
            ["stop", "length"],
            [choice.finish_reason for choice in result.choices],
        )
        self.assertEqual(["a1", "a2"], result.choices[0].logprobs.content)
        self.assertEqual(["b1", "b2"], result.choices[1].logprobs.content)
        self.assertEqual("test-model", result.model)

    async def test_sparse_choices_and_first_reasoning_are_preserved(self):
        result = await self._collect(
            _response(_choice(1, reasoning_content="reason-1")),
            _response(),
            _response(_choice(0, content="A")),
            _response(_choice(1, content="B", reasoning_content="reason-2")),
        )

        self.assertEqual([0, 1], [choice.index for choice in result.choices])
        self.assertEqual("A", result.choices[0].message.content)
        self.assertEqual("B", result.choices[1].message.content)
        self.assertEqual(
            "reason-1reason-2",
            result.choices[1].message.reasoning_content,
        )
        self.assertEqual("assistant", result.choices[1].message.role)

    async def test_negative_and_duplicate_indexes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            await self._collect(_response(_choice(-1, content="invalid")))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            await self._collect(
                _response(
                    _choice(0, content="first"),
                    _choice(0, content="second"),
                )
            )


if __name__ == "__main__":
    unittest.main()
