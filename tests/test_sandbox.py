import pytest

from sentimental_cap_predictor.sandbox import run_code


def test_attribute_access_disallowed():
    code = "__builtins__.__class__"
    with pytest.raises(ValueError):
        run_code(code)


def test_while_loop_disallowed():
    code = "while True:\n    pass"
    with pytest.raises(ValueError):
        run_code(code)


def test_loop_iteration_limit():
    code = "for i in range(20000):\n    pass"
    with pytest.raises(ValueError):
        run_code(code)


def test_dangerous_builtin_not_available():
    code = "open('x', 'w')"
    with pytest.raises(NameError):
        run_code(code)


def test_memory_limit():
    code = "x = 'a' * 200_000_000"
    with pytest.raises(RuntimeError):
        run_code(code, mem_limit=10_000_000)


def test_run_code_returns_environment():
    env = run_code("x = 1 + 2")
    assert env["x"] == 3


def test_run_code_timeout():
    code = (
        "for i in range(10000):\n"
        "    for j in range(10000):\n"
        "        pass"
    )
    with pytest.raises(TimeoutError):
        run_code(code, timeout=1, cpu_time=10)
