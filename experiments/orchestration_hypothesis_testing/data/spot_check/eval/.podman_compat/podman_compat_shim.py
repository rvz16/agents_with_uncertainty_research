
"""Monkeypatch docker SDK so SWE-bench harness can run against podman 3.4.4.

Podman 3.4.4 reports Docker compat API 1.40; the harness passes
`platform=` to client.containers.create() and client.images.build(),
which the SDK only allows on API >= 1.41. We strip `platform` from the
config dict before the SDK can complain.

Auto-no-ops on real Docker (where the platform kwarg is fine).
"""
import os
if "podman" in os.environ.get("DOCKER_HOST", "").lower() or os.environ.get("SWEBENCH_PODMAN_COMPAT"):
    import docker
    from docker.api import container as _container_api
    from docker.api import build as _build_api

    _orig_create = _container_api.ContainerApiMixin.create_container_from_config
    def _patched_create(self, config, name=None, platform=None):  # noqa: ARG001
        config.pop("platform", None)
        host_cfg = config.get("HostConfig")
        if isinstance(host_cfg, dict):
            host_cfg.pop("Platform", None)
        # Drop the explicit platform kwarg too — podman 3.4.4 reports API 1.40
        # which the SDK refuses to accept platform for.
        return _orig_create(self, config, name, None)
    _container_api.ContainerApiMixin.create_container_from_config = _patched_create  # type: ignore[assignment]

    _orig_build = _build_api.BuildApiMixin.build
    def _patched_build(self, *args, **kwargs):
        kwargs.pop("platform", None)
        return _orig_build(self, *args, **kwargs)
    _build_api.BuildApiMixin.build = _patched_build  # type: ignore[assignment]
