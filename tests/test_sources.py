import ast
import builtins
import hashlib
import io
import json
from pathlib import Path
import symtable
import tokenize
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceChecks(unittest.TestCase):
    def test_preserved_algorithm_definitions(self):
        manifest = json.loads((ROOT / "configs" / "refactor_manifest.json").read_text(encoding="utf-8"))
        for key, entry in manifest.items():
            with self.subTest(definition=key):
                path = ROOT / (entry["module"].replace(".", "/") + ".py")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                name = key.split(":", 1)[1]
                definitions = [
                    node for node in tree.body
                    if getattr(node, "name", None) == name
                    or isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name
                ]
                self.assertEqual(len(definitions), 1)
                digest = hashlib.sha256(ast.dump(definitions[0], include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest, entry["sha256"])

    def test_global_bindings(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
                bound = {
                    symbol.get_name() for symbol in table.get_symbols()
                    if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
                }
                pending = [table]
                missing = set()
                while pending:
                    current = pending.pop()
                    pending.extend(current.get_children())
                    for symbol in current.get_symbols():
                        name = symbol.get_name()
                        if symbol.is_global() and symbol.is_referenced() and name not in bound:
                            if not hasattr(builtins, name) and name not in {"__name__", "__file__"}:
                                missing.add(name)
                self.assertFalse(missing, str(missing))

    def test_package_import_graph(self):
        paths = {
            ".".join(path.relative_to(ROOT).with_suffix("").parts): path
            for path in (ROOT / "pmv_crn").rglob("*.py")
            if path.name != "__init__.py"
        }
        graph = {name: set() for name in paths}
        for name, path in paths.items():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("pmv_crn"):
                    continue
                if node.module in paths:
                    graph[name].add(node.module)
                else:
                    for alias in node.names:
                        target = node.module + "." + alias.name
                        self.assertIn(target, paths)
                        graph[name].add(target)
        complete = set()
        visiting = set()

        def visit(name):
            self.assertNotIn(name, visiting, "Circular package import: " + name)
            if name in complete:
                return
            visiting.add(name)
            for dependency in graph[name]:
                visit(dependency)
            visiting.remove(name)
            complete.add(name)

        for name in graph:
            visit(name)

    def test_syntax_and_annotation_removal(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                compile(tree, str(path), "exec")
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                self.assertFalse(any(t.type == tokenize.COMMENT for t in tokens))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node))

    def test_local_imports(self):
        for path in ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.startswith(("tmpmodel", "dual_branch_cmr")):
                        self.assertTrue((ROOT / (name + ".py")).is_file(), name)


if __name__ == "__main__":
    unittest.main()
