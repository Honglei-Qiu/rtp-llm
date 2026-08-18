import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import StreamingResponse

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.openai.api_datatype import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.misc import exception_http_status, format_exception
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    ConcurrencyException,
)


INPUT_MESSAGE = "invalid tool history"


def _input_error(message: str = INPUT_MESSAGE) -> FtRuntimeException:
    return FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, message)


def _request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], stream=stream
    )


def _response_body(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


class _AccessLogger:
    def log_query_access(self, request):
        pass

    def log_exception_access(self, request, error, formatted_error=None):
        pass


class _RawRequest:
    headers = {}

    async def is_disconnected(self):
        return False


class _FailingOpenaiEndpoint:
    def __init__(self, error):
        self.error = error

    def chat_completion(self, request_id, request, raw_request):
        raise self.error

    async def batch_chat_completion(self, request_id, request):
        raise self.error

    def chat_render(self, request):
        raise self.error

    def render_chat(self, request):
        raise self.error


class _LazyFailingOpenaiEndpoint:
    """Raises on the first generator step instead of at call time.

    Real input validation is lazy: `chat_completion` returns a generator and the error only
    surfaces on the first `__anext__`, by which point a StreamingResponse would already have
    committed HTTP 200. A synchronously-raising double takes the identical eager path for
    stream=True and stream=False, so it cannot pin the pre-header rejection at all.
    """

    def __init__(self, error):
        self.error = error

    def chat_completion(self, request_id, request, raw_request):
        error = self.error

        async def _generator():
            raise error
            yield None  # unreachable; makes this an async generator

        return CompleteResponseAsyncGenerator(_generator(), AsyncMock())


def _frontend_server(error, *, lazy: bool = False) -> FrontendServer:
    server = FrontendServer.__new__(FrontendServer)
    server.py_env_configs = SimpleNamespace(
        server_config=SimpleNamespace(ip="127.0.0.1", server_port=12345)
    )
    server.rank_id = "0"
    server.server_id = "0"
    server._access_logger = _AccessLogger()
    server._global_controller = ConcurrencyController(max_concurrency=4)
    server._frontend_worker = SimpleNamespace(
        is_streaming=lambda request: request.get("stream", False)
    )
    server._openai_endpoint = (
        _LazyFailingOpenaiEndpoint(error) if lazy else _FailingOpenaiEndpoint(error)
    )
    return server


class ExceptionHttpStatusTest(TestCase):
    def test_client_fault_categories_map_to_http_400(self):
        for exception_type in (
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            ExceptionType.INVALID_PARAMS,
            ExceptionType.LONG_PROMPT_ERROR,
            ExceptionType.UNSUPPORTED_OPERATION,
        ):
            with self.subTest(exception_type=exception_type):
                self.assertEqual(
                    exception_http_status(
                        FtRuntimeException(exception_type, "client fault")
                    ),
                    400,
                )

    def test_internal_and_raw_errors_remain_http_500(self):
        errors = (
            FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, "execution"),
            ValueError("raw value"),
            RuntimeError("raw runtime"),
            ConcurrencyException("full"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(exception_http_status(error), 500)

    def test_typed_input_payload_is_stable_and_has_no_traceback(self):
        payload = format_exception(_input_error())

        self.assertEqual(
            payload,
            {
                "error_code": 507,
                "error_code_str": "507_ERROR_INPUT_FORMAT_ERROR",
                "message": INPUT_MESSAGE,
            },
        )
        self.assertNotIn("Traceback", payload["message"])


class FrontendRendererInputErrorTest(IsolatedAsyncioTestCase):
    def assert_input_response(self, response):
        self.assertNotIsInstance(response, StreamingResponse)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            _response_body(response),
            {
                "error_code": 507,
                "error_code_str": "507_ERROR_INPUT_FORMAT_ERROR",
                "message": INPUT_MESSAGE,
            },
        )
        self.assertNotIn("text/event-stream", response.headers.get("content-type", ""))

    async def test_chat_completion_rejects_before_streaming_headers(self):
        for stream in (False, True):
            with self.subTest(stream=stream), patch(
                "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
            ):
                server = _frontend_server(_input_error())
                response = await server.chat_completion(
                    _request(stream=stream), _RawRequest()
                )

            self.assert_input_response(response)
            self.assertEqual(server._global_controller.get_available_concurrency(), 4)

    async def test_lazy_input_error_rejects_before_streaming_headers(self):
        # The error surfaces on the first generator step, which is where a StreamingResponse
        # would already have committed 200 and forced the error into an in-band frame --
        # exactly the shape that makes clients retry a request that can never succeed.
        for stream in (False, True):
            with self.subTest(stream=stream), patch(
                "rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()
            ):
                server = _frontend_server(_input_error(), lazy=True)
                response = await server.chat_completion(
                    _request(stream=stream), _RawRequest()
                )

            self.assert_input_response(response)
            self.assertEqual(server._global_controller.get_available_concurrency(), 4)

    async def test_batch_chat_completion_returns_structured_input_error(self):
        server = _frontend_server(_input_error())
        batch = BatchChatCompletionRequest(requests=[_request()])

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.batch_chat_completion(batch, _RawRequest())

        self.assert_input_response(response)
        self.assertEqual(server._global_controller.get_available_concurrency(), 4)

    async def test_chat_render_returns_structured_input_error(self):
        server = _frontend_server(_input_error())

        response = await server.chat_render(_request(), _RawRequest())

        self.assert_input_response(response)

    async def test_openai_tokenize_returns_structured_input_error(self):
        server = _frontend_server(_input_error())

        response = server.tokenize(
            {"messages": [{"role": "user", "content": "hello"}]}
        )

        self.assert_input_response(response)

    async def test_raw_renderer_value_error_remains_internal(self):
        server = _frontend_server(ValueError("renderer defect"))

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.chat_completion(_request(), _RawRequest())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(_response_body(response)["error_code"], 514)

    async def test_typed_execution_error_remains_internal(self):
        server = _frontend_server(
            FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, "renderer defect")
        )

        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            response = await server.chat_completion(_request(), _RawRequest())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            _response_body(response),
            {
                "error_code": 606,
                "error_code_str": "606_EXECUTION_EXCEPTION",
                "message": "renderer defect",
            },
        )


class OpenaiBatchInputErrorTest(IsolatedAsyncioTestCase):
    async def test_streaming_child_is_typed_and_never_enqueued(self):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.backend_rpc_server_visitor = SimpleNamespace(
            batch_enqueue=AsyncMock()
        )
        batch = BatchChatCompletionRequest(requests=[_request(stream=True)])

        with self.assertRaises(FtRuntimeException) as caught:
            await endpoint.batch_chat_completion(100, batch)

        self.assertEqual(
            caught.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertEqual(
            caught.exception.message,
            "batch chat completion does not support streaming (request index 0)",
        )
        endpoint.backend_rpc_server_visitor.batch_enqueue.assert_not_awaited()

    async def test_later_child_input_error_has_index_and_never_enqueues(self):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.backend_rpc_server_visitor = SimpleNamespace(
            batch_enqueue=AsyncMock()
        )
        first_config = SimpleNamespace(is_streaming=True)
        endpoint._prepare_chat_input = MagicMock(
            side_effect=[
                (SimpleNamespace(), first_config),
                _input_error("missing tool result"),
            ]
        )
        batch = BatchChatCompletionRequest(requests=[_request(), _request()])

        with self.assertRaises(FtRuntimeException) as caught:
            await endpoint.batch_chat_completion(100, batch)

        self.assertEqual(
            caught.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertEqual(
            caught.exception.message, "request index 1: missing tool result"
        )
        self.assertFalse(first_config.is_streaming)
        endpoint.backend_rpc_server_visitor.batch_enqueue.assert_not_awaited()

    async def test_non_input_typed_error_is_not_reclassified(self):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.backend_rpc_server_visitor = SimpleNamespace(
            batch_enqueue=AsyncMock()
        )
        error = FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, "internal")
        endpoint._prepare_chat_input = MagicMock(side_effect=error)
        batch = BatchChatCompletionRequest(requests=[_request()])

        with self.assertRaises(FtRuntimeException) as caught:
            await endpoint.batch_chat_completion(100, batch)

        self.assertIs(caught.exception, error)
        endpoint.backend_rpc_server_visitor.batch_enqueue.assert_not_awaited()


if __name__ == "__main__":
    main()
