import socket
import threading
import time

import httpx
import pytest
import uvicorn

from routewise.simulator import build_app


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="session")
def sim_url():
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(build_app(speed=0.05), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True


@pytest.fixture
def sim_config(tmp_path, sim_url):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text(f"""backends:
  - {{name: small, base_url: "{sim_url}", model: sim-small, price_in_per_m: 0.05, price_out_per_m: 0.1}}
  - {{name: medium, base_url: "{sim_url}", model: sim-medium, price_in_per_m: 0.3, price_out_per_m: 0.6}}
  - {{name: large, base_url: "{sim_url}", model: sim-large, price_in_per_m: 2.5, price_out_per_m: 10}}
""")
    return str(cfg)
