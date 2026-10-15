"""
Registry for BB (Bottleneck Benchmark) components and tasks.

Encoders and decoders self-register so configs can choose them independently.
Tasks keep the same registration pattern as before.
"""
import importlib
import pkgutil
from typing import Any

# name -> {cls, constructor_params, param_defaults}
ENCODERS: dict[str, dict[str, Any]] = {}
DECODERS: dict[str, dict[str, Any]] = {}
GENERATORS: dict[str, dict[str, Any]] = {}
# name -> {cls, description}
TASKS: dict[str, dict[str, Any]] = {}

_encoders_loaded = False
_decoders_loaded = False
_generators_loaded = False
_tasks_loaded = False


def _load_encoders():
    global _encoders_loaded
    if _encoders_loaded:
        return
    import bb.models.encoders as encoders_pkg

    for _, modname, _ in pkgutil.iter_modules(
        encoders_pkg.__path__, prefix="bb.models.encoders."
    ):
        importlib.import_module(modname)
    _encoders_loaded = True


def _load_decoders():
    global _decoders_loaded
    if _decoders_loaded:
        return
    import bb.models.decoders as decoders_pkg

    for _, modname, _ in pkgutil.iter_modules(
        decoders_pkg.__path__, prefix="bb.models.decoders."
    ):
        importlib.import_module(modname)
    _decoders_loaded = True


def _load_generators():
    global _generators_loaded
    if _generators_loaded:
        return
    import bb.generators as generators_pkg

    for _, modname, _ in pkgutil.iter_modules(
        generators_pkg.__path__, prefix="bb.generators."
    ):
        if "base" not in modname:
            importlib.import_module(modname)
    _generators_loaded = True


def _load_tasks():
    global _tasks_loaded
    if _tasks_loaded:
        return
    import bb.tasks as tasks_pkg
    for _, modname, _ in pkgutil.iter_modules(tasks_pkg.__path__, prefix="bb.tasks."):
        if "base" not in modname:
            importlib.import_module(modname)
    _tasks_loaded = True


def _component_meta(
    *,
    constructor_params: list[str] | None = None,
    param_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "constructor_params": constructor_params or [],
        "param_defaults": param_defaults or {},
    }


def register_encoder(
    name: str,
    *,
    constructor_params: list[str] | None = None,
    param_defaults: dict[str, Any] | None = None,
):
    """Register a sequence encoder."""

    def decorator(cls):
        ENCODERS[name] = {
            "cls": cls,
            **_component_meta(
                constructor_params=constructor_params,
                param_defaults=param_defaults,
            ),
        }
        return cls

    return decorator


def register_decoder(
    name: str,
    *,
    constructor_params: list[str] | None = None,
    param_defaults: dict[str, Any] | None = None,
):
    """Register a sequence decoder."""

    def decorator(cls):
        DECODERS[name] = {
            "cls": cls,
            **_component_meta(
                constructor_params=constructor_params,
                param_defaults=param_defaults,
            ),
        }
        return cls

    return decorator


def register_generator(
    name: str,
    *,
    constructor_params: list[str] | None = None,
    param_defaults: dict[str, Any] | None = None,
):
    """Register a sequence source/generator (e.g. random Markov, palindrome)."""

    def decorator(cls):
        GENERATORS[name] = {
            "cls": cls,
            **_component_meta(
                constructor_params=constructor_params,
                param_defaults=param_defaults,
            ),
        }
        return cls

    return decorator


def register_task(name: str, *, description: str = ""):
    """Register a task class. Use as @register_task('reverse', description='...')."""

    def decorator(cls):
        TASKS[name] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_encoder(name: str):
    """Get encoder class and metadata."""
    _load_encoders()
    return ENCODERS.get(name)


def get_decoder(name: str):
    """Get decoder class and metadata."""
    _load_decoders()
    return DECODERS.get(name)


def get_generator(name: str):
    """Get generator/source class and metadata."""
    _load_generators()
    return GENERATORS.get(name)


def get_task(name: str):
    """Get task class and metadata. Loads tasks on first call."""
    _load_tasks()
    return TASKS.get(name)


def all_encoder_names() -> list[str]:
    _load_encoders()
    return list(ENCODERS.keys())


def all_decoder_names() -> list[str]:
    _load_decoders()
    return list(DECODERS.keys())


def all_generator_names() -> list[str]:
    _load_generators()
    return list(GENERATORS.keys())


def all_task_names() -> list[str]:
    _load_tasks()
    return list(TASKS.keys())


def all_model_param_keys() -> set[str]:
    """Union of sweep-relevant constructor params from active encoders and decoders."""
    _load_encoders()
    _load_decoders()
    keys: set[str] = set()
    for info in ENCODERS.values():
        keys.update(info.get("constructor_params", []))
    for info in DECODERS.values():
        keys.update(info.get("constructor_params", []))
    return keys - {"vocab_size", "seq_len", "z_dim"}


def param_defaults_from_models() -> dict[str, Any]:
    """Merge param defaults from active encoders and decoders."""
    _load_encoders()
    _load_decoders()
    merged: dict[str, Any] = {}
    for info in ENCODERS.values():
        merged.update(info.get("param_defaults", {}))
    for info in DECODERS.values():
        merged.update(info.get("param_defaults", {}))
    return merged
