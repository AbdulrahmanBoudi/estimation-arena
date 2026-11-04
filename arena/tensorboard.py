"""TensorBoard at /tb, behind the same gateway check as the rest of the service.

It listens only on loopback; this forwards to it, without the sign-in and proxy headers.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

TARGET = os.environ.get("ESTIMATION_TENSORBOARD", "http://127.0.0.1:6006")
LIMIT = 20 * 1024 * 1024
DROP = {"host", "cookie", "authorization", "accept-encoding", "connection", "content-length", "transfer-encoding"}
KEEP_BACK = {"content-type", "cache-control", "etag", "last-modified"}


def _reply_headers(reply_headers) -> dict[str, str]:
    headers = {k: v for k, v in reply_headers.items() if k.lower() in KEEP_BACK}
    # Cloudflare's Rocket Loader rewrites script tags in HTML, which leaves TensorBoard's page blank.
    if reply_headers.get("content-type", "").startswith("text/html"):
        cache = headers.get("Cache-Control") or headers.get("cache-control")
        for k in ("Cache-Control", "cache-control"):
            headers.pop(k, None)
        headers["Cache-Control"] = f"{cache}, no-transform" if cache else "no-transform"
    return headers


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))


def _headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items()
            if k.lower() not in DROP and not k.lower().startswith(("remote-", "x-home-", "x-forwarded-"))}  # fmt: skip


async def root(_: Request) -> Response:
    return RedirectResponse("/tb/", status_code=308)


async def forward(request: Request) -> Response:
    body = await request.body()
    if len(body) > LIMIT:
        return PlainTextResponse("That request is too large.", status_code=413)
    path = urllib.parse.quote(request.path_params.get("path", ""), safe="/-._~")
    query = f"?{request.url.query}" if request.url.query else ""
    upstream = urllib.request.Request(f"{TARGET}/tb/{path}{query}", data=body or None, method=request.method, headers=_headers(request))
    try:
        reply = _opener.open(upstream, timeout=30)
    except urllib.error.HTTPError as error:
        reply = error
    except OSError:
        return PlainTextResponse("TensorBoard isn't running. Start the home-server-tensorboard service on the server.", status_code=502)
    with reply:
        content = reply.read(LIMIT + 1)
        headers = _reply_headers(reply.headers)
        if len(content) > LIMIT:
            return PlainTextResponse("That response is too large.", status_code=502)
        return Response(content, status_code=reply.status, headers=headers)
