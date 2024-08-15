# helper function to make a httpx client for BentoML service
def _make_httpx_client(url, svc):

    import httpx
    from urllib.parse import urlparse
    from bentoml._internal.utils.uri import uri_to_path

    timeout = svc.config["traffic"]["timeout"]
    headers = {"Runner-Name": svc.name}
    parsed = urlparse(url)
    transport = None
    target_url = url

    if parsed.scheme == "file":
        uds = uri_to_path(url)
        transport = httpx.HTTPTransport(uds=uds)
        target_url = "http://127.0.0.1:3000"
    elif parsed.scheme == "tcp":
        target_url = f"http://{parsed.netloc}"

    return httpx.Client(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    ), target_url