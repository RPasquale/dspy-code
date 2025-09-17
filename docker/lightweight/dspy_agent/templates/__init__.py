"""Template assets bundled with the DSPy agent."""

from .lightweight import (
    TemplateAsset,
    extra_lightweight_assets,
    render_compose,
    render_dockerfile,
)

__all__ = [
    "TemplateAsset",
    "extra_lightweight_assets",
    "render_compose",
    "render_dockerfile",
]
