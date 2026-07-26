"""Throwaway API-surface probe for langstage-hermes (Task 1 gating spike)."""
import inspect

import langstage_hermes as lh


def show(label, obj):
    print(f"\n== {label} ==")
    print(obj)


print("langstage_hermes version:", getattr(lh, "__version__", "?"))
print("top-level names:", sorted(n for n in dir(lh) if not n.startswith("_")))

show("create_hermes_agent signature", inspect.signature(lh.create_hermes_agent))

cfg = lh.HermesConfig
show("HermesConfig fields", getattr(cfg, "model_fields", getattr(cfg, "__annotations__", cfg)))

show("MemoryProvider ABC", [m for m in dir(lh.MemoryProvider) if not m.startswith("_")])
for m in [m for m in dir(lh.MemoryProvider) if not m.startswith("_")]:
    try:
        show(f"MemoryProvider.{m} signature", inspect.signature(getattr(lh.MemoryProvider, m)))
    except (TypeError, ValueError) as exc:
        show(f"MemoryProvider.{m} signature", f"<unavailable: {exc}>")

show("PluginContext.register_tool", inspect.signature(lh.PluginContext.register_tool))
show(
    "plugin discovery hints",
    [n for n in dir(lh) if "plugin" in n.lower()],
)
