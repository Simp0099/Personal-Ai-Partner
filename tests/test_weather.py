"""Weather tool: wttr.in primary, Google scrape fallback, honest failures.

Network is mocked throughout (no real HTTP). bs4 is stubbed where needed
so tests run without optional parsing deps.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.tools import weather  # noqa: E402


class _Resp:
    def __init__(self, text="", status_code=200):
        self.text = text
        self.status_code = status_code


@pytest.fixture()
def _quiet(monkeypatch):
    monkeypatch.setattr(weather, "speak", lambda *a: None)


def _google_html(temp="28°C"):
    return f'<html><body><div class="BNeawe iBp4i AP7Wnd">{temp}</div></body></html>'


def _bs4_stub(html):
    """Minimal BeautifulSoup stand-in parsing our canned markup."""
    import re

    class _Div:
        def __init__(self, text):
            self.text = text

    class _Soup:
        def __init__(self, text, parser=None):
            self._text = text

        def find(self, tag, class_=None, **kwargs):
            cls = class_ or kwargs.get("class_")
            m = re.search(r'<div class="([^"]*)">([^<]*)</div>', self._text)
            if m and cls and all(c in m.group(1).split() or c in m.group(1)
                                 for c in str(cls).split()):
                return _Div(m.group(2))
            return None

    mod = MagicMock()
    mod.BeautifulSoup.side_effect = lambda text, parser=None: _Soup(text)
    return mod


class TestWttrPrimary:
    def test_wttr_success(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get",
                            lambda *a, **k: _Resp("+36°C", 200))
        assert weather.get_temperature("Delhi") == "+36°C"

    def test_default_city_used(self, _quiet, monkeypatch):
        seen = []
        from jarvis.config import DEFAULT_CITY

        def _get(url, **kwargs):
            seen.append(url)
            return _Resp("+30°C", 200)

        monkeypatch.setattr(weather.requests, "get", _get)
        assert weather.get_temperature(None) == "+30°C"
        assert DEFAULT_CITY in seen[0]

    def test_malformed_wttr_falls_back(self, _quiet, monkeypatch):
        def _get(url, **kwargs):
            if "wttr.in" in url:
                return _Resp("unknown location", 200)  # no degree sign
            return _Resp(_google_html("28°C"), 200)

        monkeypatch.setattr(weather.requests, "get", _get)
        monkeypatch.setitem(sys.modules, "bs4", _bs4_stub(""))
        try:
            assert weather.get_temperature("Delhi") == "28°C"
        finally:
            monkeypatch.undo()


class TestGoogleFallback:
    def test_google_fallback_when_wttr_down(self, _quiet, monkeypatch):
        def _get(url, **kwargs):
            if "wttr.in" in url:
                raise ConnectionError("denied")
            return _Resp(_google_html("28°C"), 200)

        monkeypatch.setattr(weather.requests, "get", _get)
        monkeypatch.setitem(sys.modules, "bs4", _bs4_stub(""))
        try:
            assert weather.get_temperature("Delhi") == "28°C"
        finally:
            monkeypatch.undo()

    def test_both_fail_honest_empty(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get",
                            MagicMock(side_effect=ConnectionError("down")))
        spoken = []
        monkeypatch.setattr(weather, "speak", lambda msg: spoken.append(msg))
        assert weather.get_temperature("Delhi") == ""
        assert any("Unable" in msg for msg in spoken)
