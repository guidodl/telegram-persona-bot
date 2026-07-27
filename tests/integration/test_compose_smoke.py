import subprocess

import pytest


@pytest.mark.slow
def test_compose_config_is_valid():
    r = subprocess.run(["docker", "compose", "config"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "pgvector/pgvector:pg16" in r.stdout
