import pytest

from celnn.core.exceptions import CelNNError
from celnn.templates import Template, TemplateRegistry


def test_registry_register_get_remove():
    registry = TemplateRegistry()
    template = Template(
        name="demo", feedback=[0.0, 1.0, 0.0], control=[0.0, 0.0, 0.0]
    )
    registry.register(template)
    assert registry.names() == ["demo"]
    restored = registry.get("demo")
    assert restored.name == "demo"
    removed = registry.remove("demo")
    assert removed.name == "demo"
    assert registry.names() == []


def test_registry_rejects_duplicate_without_overwrite():
    registry = TemplateRegistry()
    template = Template(
        name="demo", feedback=[0.0, 1.0, 0.0], control=[0.0, 0.0, 0.0]
    )
    registry.register(template)
    with pytest.raises(CelNNError):
        registry.register(template)
