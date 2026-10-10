from unittest.mock import patch

from jarvis.tools import web
from jarvis.tools import system_tools


def test_known_service_uses_canonical_url():
    with patch.object(web.webbrowser, "open") as open_url:
        assert web.open_service("open YouTube website") == "Opened https://www.youtube.com"
    open_url.assert_called_once_with("https://www.youtube.com")


def test_unknown_service_does_not_construct_or_open_a_domain():
    with patch.object(web.webbrowser, "open") as open_url:
        result = web.open_service("attacker.example")
    assert "trusted URL" in result
    open_url.assert_not_called()


def test_tool_web_actions_do_not_speak_outside_the_conversation_layer():
    import inspect

    assert "speak(" not in inspect.getsource(web)


def test_no_registered_tool_module_speaks_or_blocks_for_input():
    """Tools return text; the conversation layer owns speech and questions.

    A tool that speaks talks over the reply, and a tool that calls listen()
    blocks the turn on a microphone the user never opened. Both are checked
    here over the source of every registered tool module, so a new tool cannot
    reintroduce either without failing a test.
    """
    import inspect
    import pathlib

    import pathlib

    from jarvis.tools import apps  # any module in the tools package

    tools_dir = pathlib.Path(inspect.getfile(apps)).parent

    for module_path in sorted(tools_dir.glob("*.py")):
        source = inspect.cleandoc(module_path.read_text(encoding="utf-8"))
        # Skip the definition line of an import or the module docstring.
        assert "speak(" not in source, f"{module_path.name} speaks"
        assert "listen(" not in source, f"{module_path.name} blocks on input"


def test_directory_listing_denies_paths_outside_configured_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setattr(system_tools, "ALLOWED_FILE_ROOTS", (allowed.resolve(),))
    result = system_tools.get_directory_contents(str(outside))
    assert "Access denied" in result


def test_directory_listing_skips_symlink_escape(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "credential.txt").write_text("secret payload")
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(system_tools, "ALLOWED_FILE_ROOTS", (allowed.resolve(),))
    result = system_tools.get_directory_contents(str(allowed))
    assert "credential" not in result
    assert "secret payload" not in result


def test_tool_logs_omit_argument_and_result_values(caplog):
    import logging
    from jarvis.logger import StatusIndicator

    caplog.set_level(logging.DEBUG, logger="jarvis")
    StatusIndicator.tool_call("send_email", {"content": "private email body"})
    StatusIndicator.tool_result("recall_memories", "private memory fact")
    assert "private email body" not in caplog.text
    assert "private memory fact" not in caplog.text
    assert "send_email" in caplog.text
