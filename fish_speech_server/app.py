import re
import warnings
from threading import Lock

# Suppress torch.compile (Inductor) spam: "Logical operators 'and'/'or' deprecated, use '&'/'|'"
# Source: torch._inductor.runtime.triton_helpers — fix belongs in PyTorch upstream
warnings.filterwarnings(
    "ignore",
    message=".*Logical operators 'and' and 'or' are deprecated for non-scalar tensors.*",
    category=UserWarning,
    module="torch._inductor",
)

import pyrootutils
import uvicorn
from kui.asgi import Depends, FactoryClass, HTTPException, Kui, OpenAPI, Routes
from kui.cors import CORSConfig
from kui.openapi.specification import Info
from kui.security import bearer_auth
from loguru import logger
from typing_extensions import Annotated

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech_server.api.utils import MsgPackRequest, parse_args
from fish_speech_server.api.exception_handler import ExceptionHandler
from fish_speech_server.services.model_manager import ModelManager
from fish_speech_server.api.views import routes


class API(ExceptionHandler):
    def __init__(self):
        self.args = parse_args()

        def api_auth(endpoint):
            async def verify(token: Annotated[str, Depends(bearer_auth)]):
                if token != self.args.api_key:
                    raise HTTPException(401, None, "Invalid token")
                return await endpoint()

            async def passthrough():
                return await endpoint()

            if self.args.api_key is not None:
                return verify
            else:
                return passthrough

        self.routes = Routes(
            routes,  # keep existing routes
            http_middlewares=[api_auth],  # apply api_auth middleware
        )

        # OpenAPIの設定
        self.openapi = OpenAPI(
            Info(
                {
                    "title": "Fish Speech API",
                    "version": "1.5.0",
                }
            ),
        ).routes

        # Initialize the app
        self.app = Kui(
            routes=self.routes + self.openapi[1:],  # Remove the default route
            exception_handlers={
                HTTPException: self.http_exception_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config=CORSConfig(),
        )

        # Add the state variables
        self.app.state.lock = Lock()
        self.app.state.device = self.args.device
        self.app.state.max_text_length = self.args.max_text_length

        # Associate the app with the model manager
        self.app.on_startup(self.initialize_app)

    async def initialize_app(self, app: Kui):
        # Make the ModelManager available to the views
        app.state.model_manager = ModelManager(
            mode=self.args.mode,
            device=self.args.device,
            half=self.args.half,
            compile=self.args.compile,
            llama_checkpoint_path=self.args.llama_checkpoint_path,
            decoder_checkpoint_path=self.args.decoder_checkpoint_path,
            decoder_config_name=self.args.decoder_config_name,
        )

        logger.info(f"Startup done, listening server at http://{self.args.listen}")


def split_listen_address(listen: str) -> tuple[str, int]:
    # IPv6 address format is [xxxx:xxxx::xxxx]:port
    match = re.search(r"\[([^\]]+)\]:(\d+)$", listen)
    if match:
        host, port = match.groups()  # IPv6
    else:
        host, port = listen.split(":")  # IPv4
    return host, int(port)


def create_app() -> Kui:
    return API().app


def run_api(api: API | None = None) -> None:
    api = api or API()
    host, port = split_listen_address(api.args.listen)

    uvicorn.run(
        api.app,
        host=host,
        port=port,
        workers=api.args.workers,
        log_level="info",
    )


def main() -> None:
    # Each worker process created by Uvicorn has its own memory space,
    # so model handles are not shared across workers.
    run_api()


if __name__ == "__main__":
    main()
