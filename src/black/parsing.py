"""
Parse Python code and perform AST validation.
"""

import ast
import sys
import warnings
from collections.abc import Collection, Iterator

from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node


class InvalidInput(ValueError):
      """
    功能：自定义异常类，当输入源代码无法被任何版本的语法解析器解析时抛出
    应用场景：lib2to3_parse 函数中，所有尝试的语法规则都解析失败时触发
    继承关系：继承自 ValueError，属于业务逻辑异常，非系统级错误
    """


def get_grammars(target_versions: set[TargetVersion]) -> list[Grammar]:
    """
       功能：根据目标Python版本，返回对应的语法解析规则（Grammar）列表
       核心逻辑：不同Python版本的语法规则存在差异（如async关键字、模式匹配），需匹配对应规则
       入参：target_versions - 目标Python版本集合（如{TargetVersion.PY310, TargetVersion.PY39}）
       返回值：适配的Grammar对象列表，按解析优先级排序

       版本适配逻辑：
       1. 未指定目标版本：返回全量语法规则（3.0-3.6、3.7-3.9、3.10+）
       2. 指定版本：
          - 3.7-3.9：使用python_grammar_async_keywords（async为关键字）
          - 3.0-3.6：使用python_grammar（async为标识符）
          - 3.10+：使用python_grammar_soft_keywords（支持模式匹配/软关键字）
       """
    if not target_versions:
        # No target_version specified, so try all grammars.
        return [
            # Python 3.7-3.9
            pygram.python_grammar_async_keywords,
            # Python 3.0-3.6
            pygram.python_grammar,
            # Python 3.10+
            pygram.python_grammar_soft_keywords,
        ]

    grammars = []
    # If we have to parse both, try to parse async as a keyword first
    if not supports_feature(
        target_versions, Feature.ASYNC_IDENTIFIERS
    ) and not supports_feature(target_versions, Feature.PATTERN_MATCHING):
        # Python 3.7-3.9
        grammars.append(pygram.python_grammar_async_keywords)
    if not supports_feature(target_versions, Feature.ASYNC_KEYWORDS):
        # Python 3.0-3.6
        grammars.append(pygram.python_grammar)
    if any(Feature.PATTERN_MATCHING in VERSION_TO_FEATURES[v] for v in target_versions):
        # Python 3.10+
        grammars.append(pygram.python_grammar_soft_keywords)

    # At least one of the above branches must have been taken, because every Python
    # version has exactly one of the two 'ASYNC_*' flags
    return grammars


def lib2to3_parse(
    src_txt: str, target_versions: Collection[TargetVersion] = ()
) -> Node:
    """功能：将Python源代码字符串解析为lib2to3库的Node（AST根节点）
    入参：
      - src_txt：待解析的源代码字符串
      - target_versions：目标Python版本集合（可选，默认适配所有版本）
    返回值：lib2to3.pytree.Node - AST抽象语法树根节点
    异常：InvalidInput - 所有语法规则均解析失败时抛出

    执行流程：
    1. 补全源代码末尾换行（避免解析器因无终止符报错）
    2. 获取适配目标版本的语法规则列表
    3. 遍历语法规则，尝试解析源代码：
       - 解析成功：返回AST节点
       - 解析失败：记录错误信息（行号、列号、错误行内容）
    4. 所有规则解析失败：抛出最新版本的解析错误"""
    if not src_txt.endswith("\n"):
        src_txt += "\n"

    grammars = get_grammars(set(target_versions))
    if target_versions:
        max_tv = max(target_versions, key=lambda tv: tv.value)
        tv_str = f" for target version {max_tv.pretty()}"
    else:
        tv_str = ""

    errors = {}
    for grammar in grammars:
        drv = driver.Driver(grammar)
        try:
            result = drv.parse_string(src_txt, False)
            break

        except ParseError as pe:
            lineno, column = pe.context[1]
            lines = src_txt.splitlines()
            try:
                faulty_line = lines[lineno - 1]
            except IndexError:
                faulty_line = "<line number missing in source>"
            errors[grammar.version] = InvalidInput(
                f"Cannot parse{tv_str}: {lineno}:{column}: {faulty_line}"
            )

        except TokenError as te:
            # In edge cases these are raised; and typically don't have a "faulty_line".
            lineno, column = te.args[1]
            errors[grammar.version] = InvalidInput(
                f"Cannot parse{tv_str}: {lineno}:{column}: {te.args[0]}"
            )

    else:
        # Choose the latest version when raising the actual parsing error.
        assert len(errors) >= 1
        exc = errors[max(errors)]
        raise exc from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])
    return result


def matches_grammar(src_txt: str, grammar: Grammar) -> bool:
    """
       功能：校验源代码是否符合指定的语法规则
       入参：
         - src_txt：待校验的源代码字符串
         - grammar：目标语法规则（Grammar对象）
       返回值：bool - 符合返回True，不符合返回False
       异常处理：捕获解析/词法/缩进错误，统一返回False（非抛出异常）
       """
    drv = driver.Driver(grammar)
    try:
        drv.parse_string(src_txt, False)
    except (ParseError, TokenError, IndentationError):
        return False
    else:
        return True


def lib2to3_unparse(node: Node) -> str:
    """
    功能：将lib2to3的AST节点转换回源代码字符串
    入参：node - lib2to3.pytree.Node（AST节点）
    返回值：str - 反解析后的源代码字符串
    应用场景：格式化后AST节点转回代码、调试AST结构
    """
    code = str(node)
    return code


class ASTSafetyError(Exception):
    """Raised when Black's generated code is not equivalent to the old AST."""


def _parse_single_version(
    src: str, version: tuple[int, int], *, type_comments: bool
) -> ast.AST:
    filename = "<unknown>"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(
            src, filename, feature_version=version, type_comments=type_comments
        )


def parse_ast(src: str) -> ast.AST:
    """
       功能：将 Python 源代码解析为 AST（抽象语法树）
       依赖工具：blib2to3（Python 官方语法解析库，用于兼容不同版本 Python 语法）
       输出结果：pytree.Node 类型的 AST 根节点，包含代码的语法结构信息
       应用场景：格式化流程的第一步，为后续代码重构提供语法结构支撑
       """
    # TODO: support Python 4+ ;)
    versions = [(3, minor) for minor in range(3, sys.version_info[1] + 1)]

    first_error = ""
    for version in sorted(versions, reverse=True):
        try:
            return _parse_single_version(src, version, type_comments=True)
        except SyntaxError as e:
            if not first_error:
                first_error = str(e)

    # Try to parse without type comments
    for version in sorted(versions, reverse=True):
        try:
            return _parse_single_version(src, version, type_comments=False)
        except SyntaxError:
            pass

    raise SyntaxError(first_error)


def _normalize(lineend: str, value: str) -> str:
    # To normalize, we strip any leading and trailing space from
    # each line...
    stripped: list[str] = [i.strip() for i in value.splitlines()]
    normalized = lineend.join(stripped)
    # ...and remove any blank lines at the beginning and end of
    # the whole string
    return normalized.strip()


def stringify_ast(node: ast.AST) -> Iterator[str]:
    """Simple visitor generating strings to compare ASTs by content."""
    return _stringify_ast(node, [])


def _stringify_ast_with_new_parent(
    node: ast.AST, parent_stack: list[ast.AST], new_parent: ast.AST
) -> Iterator[str]:
    parent_stack.append(new_parent)
    yield from _stringify_ast(node, parent_stack)
    parent_stack.pop()


def _stringify_ast(node: ast.AST, parent_stack: list[ast.AST]) -> Iterator[str]:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.kind == "u"
    ):
        # It's a quirk of history that we strip the u prefix over here. We used to
        # rewrite the AST nodes for Python version compatibility and we never copied
        # over the kind
        node.kind = None

    yield f"{'    ' * len(parent_stack)}{node.__class__.__name__}("

    for field in sorted(node._fields):
        # TypeIgnore has only one field 'lineno' which breaks this comparison
        if isinstance(node, ast.TypeIgnore):
            break

        try:
            value: object = getattr(node, field)
        except AttributeError:
            continue

        yield f"{'    ' * (len(parent_stack) + 1)}{field}="

        if isinstance(value, list):
            for item in value:
                # Ignore nested tuples within del statements, because we may insert
                # parentheses and they change the AST.
                if (
                    field == "targets"
                    and isinstance(node, ast.Delete)
                    and isinstance(item, ast.Tuple)
                ):
                    for elt in _unwrap_tuples(item):
                        yield from _stringify_ast_with_new_parent(
                            elt, parent_stack, node
                        )

                elif isinstance(item, ast.AST):
                    yield from _stringify_ast_with_new_parent(item, parent_stack, node)

        elif isinstance(value, ast.AST):
            yield from _stringify_ast_with_new_parent(value, parent_stack, node)

        else:
            normalized: object
            if (
                isinstance(node, ast.Constant)
                and field == "value"
                and isinstance(value, str)
                and len(parent_stack) >= 2
                # Any standalone string, ideally this would
                # exactly match black.nodes.is_docstring
                and isinstance(parent_stack[-1], ast.Expr)
            ):
                # Constant strings may be indented across newlines, if they are
                # docstrings; fold spaces after newlines when comparing. Similarly,
                # trailing and leading space may be removed.
                normalized = _normalize("\n", value)
            elif field == "type_comment" and isinstance(value, str):
                # Trailing whitespace in type comments is removed.
                normalized = value.rstrip()
            else:
                normalized = value
            yield (
                f"{'    ' * (len(parent_stack) + 1)}{normalized!r},  #"
                f" {value.__class__.__name__}"
            )

    yield f"{'    ' * len(parent_stack)})  # /{node.__class__.__name__}"


def _unwrap_tuples(node: ast.Tuple) -> Iterator[ast.AST]:
    """
        功能：递归展开嵌套的元组AST节点
        入参：node - ast.Tuple节点（可能嵌套）
        返回值：迭代器，生成所有嵌套层级的元组元素
        应用场景：del语句中的元组处理，忽略括号添加导致的AST结构变化
        """
    for elt in node.elts:
        if isinstance(elt, ast.Tuple):
            yield from _unwrap_tuples(elt)
        else:
            yield elt
