"""NASA APOD tool: honest fetching, honest dates, bounded summaries.

Network is mocked. The critical property is that a failed fetch returns no
data rather than a plausible-looking summary of a picture nobody fetched.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.tools import nasa  # noqa: E402


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


APOD_OK = {
    "title": "The Carina Nebula",
    "date": "2026-01-02",
    "explanation": "A stellar nursery. " "It spans seven light years. " "It is very young.",
    "media_type": "image",
    "url": "https://example.com/carina.jpg",
}


@pytest.fixture()
def _ok(monkeypatch):
    monkeypatch.setattr(nasa.requests, "get", lambda *a, **k: _Resp(APOD_OK))


class TestFetch:
    def test_returns_payload(self, _ok):
        assert nasa.get_nasa_apod("2026-01-02")["title"] == "The Carina Nebula"

    def test_http_error_returns_empty(self, monkeypatch):
        monkeypatch.setattr(nasa.requests, "get",
                            lambda *a, **k: _Resp({"error": {}}, 404))
        assert nasa.get_nasa_apod("1999-01-01") == {}

    def test_network_error_returns_empty(self, monkeypatch):
        def _boom(*a, **k):
            raise ConnectionError("down")

        monkeypatch.setattr(nasa.requests, "get", _boom)
        assert nasa.get_nasa_apod() == {}

    def test_timeout_is_passed(self, monkeypatch):
        seen = {}

        def _get(url, **kwargs):
            seen.update(kwargs)
            return _Resp(APOD_OK)

        monkeypatch.setattr(nasa.requests, "get", _get)
        nasa.get_nasa_apod()
        assert seen["timeout"] > 0


class TestDates:
    def test_iso_passthrough(self):
        assert nasa.normalize_spoken_date("2024-03-15") == "2024-03-15"

    def test_spaced_date(self):
        assert nasa.normalize_spoken_date("2024 03 15") == "2024-03-15"

    def test_slashed_date(self):
        assert nasa.normalize_spoken_date("2024/03/15") == "2024-03-15"

    def test_today(self):
        import datetime
        assert nasa.normalize_spoken_date("today") == datetime.date.today().isoformat()

    def test_garbage_falls_back_to_today(self):
        import datetime
        assert nasa.normalize_spoken_date("sometime last week") == datetime.date.today().isoformat()

    def test_empty_falls_back_to_today(self):
        import datetime
        assert nasa.normalize_spoken_date("") == datetime.date.today().isoformat()


class TestFormatting:
    def test_reports_the_date_nasa_actually_served(self):
        """A date with no picture returns NASA's latest, so the requested date
        would be a lie."""
        out = nasa.format_apod(APOD_OK)
        assert "2026-01-02" in out
        assert "Carina" in out
        assert "example.com/carina.jpg" in out

    def test_summary_is_bounded(self):
        summary = nasa.summarize_apod(APOD_OK, max_sentences=2)
        assert summary.count(".") == 2
        assert "very young" not in summary

    def test_summary_falls_back_to_caption(self):
        out = nasa.summarize_apod({"title": "T", "media_type": "video",
                                   "caption": "A flythrough"})
        assert "flythrough" in out

    def test_empty_payload_is_honest(self):
        assert "not responding" in nasa.format_apod({})
