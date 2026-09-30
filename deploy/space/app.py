"""Hugging Face Space entry point: the BirdID API, served on port 7860.

This runs the same FastAPI app that ./dev.sh runs locally — nothing forked.
The Space uses the Gradio SDK on ZeroGPU hardware because that is the free
tier; the API itself never needs the GPU, so the device is pinned to CPU
before bird_core picks one (outside @spaces.GPU functions ZeroGPU emulates
CUDA, which would otherwise look like a real GPU).

The website lives on Vercel and reaches this API through a rewrite of /api.
Opening the Space directly shows a one-line page pointing there.
"""

import os
import threading

os.environ.setdefault("BIRD_DEVICE", "cpu")

import spaces  # noqa: E402  — ZeroGPU expects this import at startup
import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from api import app  # noqa: E402
from services import verifier  # noqa: E402


@spaces.GPU(duration=5)
def _gpu_noop():
    """Never called. ZeroGPU will not start a Space without one GPU function."""
    return True


def _report_to_zerogpu():
    """Tell ZeroGPU which GPU functions exist.

    `spaces` normally does this from inside gr.Blocks.launch(), which it patches.
    This Space serves a FastAPI app with a Gradio page mounted on it — the
    documented way to combine the two — so launch() is never called and the
    report would never be sent; ZeroGPU then refuses to start the Space. Run the
    same startup hook directly instead. Absent outside ZeroGPU, so a no-op there.
    """
    startup = getattr(getattr(spaces, "zero", None), "startup", None)
    if startup is not None:
        startup()


_report_to_zerogpu()

SITE_URL = os.environ.get("BIRDID_SITE_URL", "").strip()

with gr.Blocks(title="BirdID API") as info_page:
    gr.Markdown(
        "# BirdID API\n"
        "This Space runs the backend for BirdID: bird identification by photo and "
        "by call, the field guide, the deck and the chat.\n\n"
        + (f"**Open the site: [{SITE_URL}]({SITE_URL})**\n\n" if SITE_URL else "")
        + "API schema: [/docs](/docs) · health: [/api/health](/api/health)"
    )

app = gr.mount_gradio_app(app, info_page, path="/gradio")
app.add_api_route("/", lambda: RedirectResponse("/gradio/"), include_in_schema=False)


def _warm_verifier():
    # BioCLIP downloads (~400 MB) and loads on first use; do it in the
    # background at boot so the first visitor's outside-the-200 photo is not
    # the one that pays for it.
    try:
        verifier.warm(progress=False)
    except Exception as e:  # never let a warm-up take the API down
        print(f"verifier warm-up skipped: {type(e).__name__}: {e}")


threading.Thread(target=_warm_verifier, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
