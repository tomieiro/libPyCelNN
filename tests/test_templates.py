import pytest

from celnn.core.exceptions import TemplateValidationError
from celnn.templates import Template


def test_template_validation_passes_for_valid_template():
    template = Template(
        name="valid", feedback=[0.0, 1.0, 0.0], control=[0.0, 0.0, 0.0]
    )
    assert template.validate() is template


def test_template_validation_rejects_even_shape():
    template = Template(
        name="invalid", feedback=[1.0, 0.0], control=[0.0, 0.0]
    )
    with pytest.raises(TemplateValidationError):
        template.validate()


def test_template_copy_and_with_bias():
    template = Template(
        name="valid",
        feedback=[0.0, 1.0, 0.0],
        control=[0.0, 0.0, 0.0],
        bias=0.0,
    )
    copy = template.copy()
    biased = template.with_bias(-0.5)
    assert copy.name == template.name
    assert biased.bias == -0.5
