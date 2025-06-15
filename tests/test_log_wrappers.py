from pathlib import Path
import ast

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_public_functions_are_decorated() -> None:
    for py_file in SRC_DIR.rglob("*.py"):
        if py_file.name == "logging.py":
            # functions in this module implement the decorator itself
            # and are excluded from decoration checks
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and not node.name.startswith("_")
            ):
                has_decorator = any(
                    isinstance(d, ast.Name) and d.id == "log_call" or
                    isinstance(d, ast.Attribute) and d.attr == "log_call"
                    for d in node.decorator_list
                )
                assert has_decorator, (
                    f"{py_file}:{node.name} missing @log_call"
                )
