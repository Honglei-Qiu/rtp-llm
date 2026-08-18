from types import SimpleNamespace
from unittest import TestCase, main

from jinja2 import TemplateError, TemplateSyntaxError

from rtp_llm.config.exceptions import (
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint

RENDER_MESSAGE = "Invalid request: Incorrect role assistant."
TEMPLATE_MESSAGE = "Conversation roles must alternate user/assistant"


class _Renderer:
    def __init__(self, error=None, rendered_prompt="hello"):
        self.error = error
        self.rendered_prompt = rendered_prompt
        self.calls = 0

    def render_chat(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            rendered_prompt=self.rendered_prompt,
            input_ids=[1, 2, 3],
            multimodal_inputs=None,
        )


class _Tokenizer:
    def encode(self, text):
        return [9] * len(text)


def _endpoint(renderer):
    endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
    endpoint.chat_renderer = renderer
    endpoint.template_renderer = renderer
    endpoint.tokenizer = _Tokenizer()
    return endpoint


def _request(partial: bool = False, user_template: str = "") -> ChatCompletionRequest:
    messages = [{"role": "user", "content": "hi"}]
    if partial:
        messages.append({"role": "assistant", "content": "draft", "partial": True})
    request = ChatCompletionRequest(messages=messages)
    if user_template:
        request.user_template = user_template
    return request


class RenderInputErrorTest(TestCase):
    def test_render_chat_reports_malformed_input_as_client_fault(self):
        endpoint = _endpoint(_Renderer(error=ValueError(RENDER_MESSAGE)))
        with self.assertRaises(FtRuntimeException) as caught:
            endpoint.render_chat(_request())
        error = caught.exception
        self.assertEqual(error.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR)
        self.assertEqual(
            error.exception_type.category, ExceptionCategory.BAD_REQUEST
        )
        self.assertIn(RENDER_MESSAGE, error.message)
        self.assertIsInstance(error.__cause__, ValueError)

    def test_chat_render_debug_path_reports_the_same_way(self):
        endpoint = _endpoint(_Renderer(error=ValueError(RENDER_MESSAGE)))
        with self.assertRaises(FtRuntimeException) as caught:
            endpoint.chat_render(_request())
        self.assertEqual(
            caught.exception.exception_type.category, ExceptionCategory.BAD_REQUEST
        )

    def test_type_error_still_surfaces_as_a_server_fault(self):
        # A TypeError means one of our own calls was shaped wrongly; translating it would
        # report an internal incident as a caller mistake and hide it.
        endpoint = _endpoint(_Renderer(error=TypeError("render_chat() takes 1 arg")))
        with self.assertRaises(TypeError):
            endpoint.render_chat(_request())

    def test_template_rejection_is_a_client_fault(self):
        # HuggingFace chat templates reject an unusable conversation with raise_exception(),
        # which arrives as TemplateError, not ValueError.
        endpoint = _endpoint(_Renderer(error=TemplateError(TEMPLATE_MESSAGE)))
        with self.assertRaises(FtRuntimeException) as caught:
            endpoint.render_chat(_request())
        self.assertEqual(
            caught.exception.exception_type.category, ExceptionCategory.BAD_REQUEST
        )
        self.assertIn(TEMPLATE_MESSAGE, caught.exception.message)

    def test_broken_shipped_template_stays_a_server_fault(self):
        endpoint = _endpoint(_Renderer(error=TemplateSyntaxError("bad", 1)))
        with self.assertRaises(TemplateSyntaxError):
            endpoint.render_chat(_request())

    def test_broken_caller_supplied_template_is_a_client_fault(self):
        endpoint = _endpoint(_Renderer(error=TemplateSyntaxError("bad", 1)))
        with self.assertRaises(FtRuntimeException) as caught:
            endpoint.render_chat(_request(user_template="{% if %}"))
        self.assertEqual(
            caught.exception.exception_type.category, ExceptionCategory.BAD_REQUEST
        )

    def test_successful_render_is_unchanged(self):
        renderer = _Renderer()
        rendered = _endpoint(renderer).render_chat(_request())
        self.assertEqual(rendered.input_ids, [1, 2, 3])
        self.assertEqual(rendered.rendered_prompt, "hello")
        self.assertEqual(renderer.calls, 1)

    def test_partial_message_prepopulation_is_unchanged(self):
        renderer = _Renderer()
        request = _request(partial=True)
        rendered = _endpoint(renderer).render_chat(request)
        # The partial trailing message is popped and appended to the rendered prompt.
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(rendered.rendered_prompt, "hello" + "draft")
        self.assertEqual(rendered.input_ids, [1, 2, 3] + [9] * len("draft"))

    def test_no_render_call_bypasses_the_translation(self):
        # Both entry points must go through _render_chat_input; a new call site that talks
        # to the renderer directly would silently reintroduce the 5xx.
        import inspect

        import rtp_llm.openai.openai_endpoint as module

        source = inspect.getsource(module)
        direct = [
            line.strip()
            for line in source.splitlines()
            if "renderer.render_chat(" in line
        ]
        self.assertEqual(len(direct), 1, direct)
        helper = inspect.getsource(OpenaiEndpoint._render_chat_input)
        self.assertIn("renderer.render_chat(", helper)


if __name__ == "__main__":
    main()
