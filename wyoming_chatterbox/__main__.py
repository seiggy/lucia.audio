#!/usr/bin/env python3
"""Main entry point for the Wyoming Chatterbox TTS server."""

import argparse
import asyncio
import logging
import signal
from functools import partial
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .handler import ChatterboxEventHandler
from .tts_engine import TTSEngine
from .voice_manager import VoiceManager

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox TTS Server")
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10200",
        help="Wyoming server URI (default: tcp://0.0.0.0:10200)",
    )
    parser.add_argument(
        "--voices-dir",
        default="/data/voices",
        help="Directory for voice profiles (default: /data/voices)",
    )
    parser.add_argument(
        "--default-voice",
        default=None,
        help="Default voice profile name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable fp16 inference",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8095,
        help="Dashboard web UI port (default: 8095, 0 to disable)",
    )
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="chatterbox",
        help="Enable zeroconf discovery (default name: chatterbox)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-format",
        default="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        help="Log format",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=args.log_format,
    )

    # Initialize voice manager
    voice_manager = VoiceManager(args.voices_dir)

    # Initialize and load TTS engine
    engine = TTSEngine(device=args.device, half_precision=not args.no_half)
    await engine.load_model()

    # Compute conditionals for any profiles that don't have them yet
    for profile in voice_manager.list_profiles():
        if not profile.is_ready:
            audio_path = voice_manager.get_audio_path(profile.id)
            conds_path = voice_manager.get_profile_dir(profile.id) / "conds.pt"
            if audio_path and audio_path.exists():
                try:
                    await engine.compute_conditionals_async(
                        str(audio_path), str(conds_path)
                    )
                    voice_manager.mark_ready(profile.id)
                    _LOGGER.info("Computed conditionals for profile: %s", profile.name)
                except Exception as e:
                    _LOGGER.error("Failed to compute conditionals for %s: %s", profile.name, e)

    # Build Wyoming info with available voices
    voices = []
    for profile in voice_manager.list_profiles():
        if profile.is_ready:
            voices.append(
                TtsVoice(
                    name=profile.name,
                    description=profile.description or f"Chatterbox voice: {profile.name}",
                    attribution=Attribution(
                        name="Resemble AI",
                        url="https://github.com/resemble-ai/chatterbox",
                    ),
                    installed=True,
                    version=__version__,
                    languages=["en"],
                )
            )

    # Always include a default entry
    if not voices:
        voices.append(
            TtsVoice(
                name="default",
                description="Chatterbox Turbo default voice",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox",
                ),
                installed=True,
                version=__version__,
                languages=["en"],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="chatterbox",
                description="Chatterbox Turbo TTS by Resemble AI — zero-shot voice cloning",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox",
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=__version__,
            )
        ],
    )

    # Start dashboard if enabled
    dashboard_task = None
    if args.dashboard_port > 0:
        from dashboard.app import create_app

        app = create_app(engine, voice_manager, wyoming_info)
        dashboard_task = asyncio.create_task(
            _run_dashboard(app, args.dashboard_port)
        )
        _LOGGER.info("Dashboard available at http://0.0.0.0:%d", args.dashboard_port)

    # Start Wyoming server
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// URI")
        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.info("Zeroconf discovery enabled as '%s'", args.zeroconf)

    _LOGGER.info("Wyoming Chatterbox TTS server ready on %s", args.uri)

    server_task = asyncio.create_task(
        server.run(
            partial(
                ChatterboxEventHandler,
                wyoming_info,
                engine,
                voice_manager,
                args.default_voice,
            )
        )
    )

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, server_task.cancel)
    loop.add_signal_handler(signal.SIGTERM, server_task.cancel)

    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("Server stopped")
    finally:
        if dashboard_task:
            dashboard_task.cancel()
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass


async def _run_dashboard(app, port: int) -> None:
    """Run the FastAPI dashboard with uvicorn."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
