"""Weather tool: Open-Meteo geocoding + forecast, honest failures.

Network is mocked throughout, so no test touches the real API. The point of
these tests is that the tool reports what it actually got -- never a
temperature it made up when a lookup failed.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.tools import weather  # noqa: E402


class _Resp:
    """Minimal requests.Response stand-in returning canned JSON."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


GEOCODE_OK = {"results": [{"latitude": 28.61, "longitude": 77.21, "name": "Delhi"}]}
FORECAST_OK = {
    "current": {
        "temperature_2m": 36.0,
        "relative_humidity_2m": 48,
        "apparent_temperature": 41.0,
        "weather_code": 0,
    },
    "daily": {
        "temperature_2m_max": [39.0],
        "temperature_2m_min": [27.0],
        "precipitation_probability_max": [5],
    },
}


def _route(geocode=None, forecast=None):
    """Build a requests.get stand-in keyed on which URL is called."""

    def _get(url, **kwargs):
        payload = geocode if "geocoding" in url else forecast
        if isinstance(payload, Exception):
            raise payload
        return _Resp(payload if payload is not None else {})

    return _get


@pytest.fixture()
def _quiet():
    """The weather tool takes no speech dependency at all.

    Asserting that here rather than patching a name in: if the module ever
    imports `speak` again, the tests fail to collect instead of quietly
    passing a stub nobody installed.
    """
    import inspect

    assert "speak" not in inspect.getsource(weather)


class TestWeatherLookup:
    def test_reports_temperature_conditions_and_range(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get",
                            _route(GEOCODE_OK, FORECAST_OK))
        out = weather.get_weather("Delhi")
        assert "Delhi" in out
        assert "36" in out
        assert "clear" in out
        assert "27" in out and "39" in out

    def test_default_city_used_when_omitted(self, _quiet, monkeypatch):
        from jarvis.config import DEFAULT_CITY

        seen = []

        def _get(url, **kwargs):
            seen.append(kwargs.get("params", {}))
            return _Resp(GEOCODE_OK if "geocoding" in url else FORECAST_OK)

        monkeypatch.setattr(weather.requests, "get", _get)
        weather.get_weather(None)
        assert seen[0].get("name") == DEFAULT_CITY

    def test_rain_advice_appears_when_wet(self, _quiet, monkeypatch):
        wet = {
            "current": {"temperature_2m": 22.0, "apparent_temperature": 22.0,
                        "weather_code": 63},
            "daily": {"temperature_2m_max": [24.0], "temperature_2m_min": [19.0],
                      "precipitation_probability_max": [80]},
        }
        monkeypatch.setattr(weather.requests, "get",
                            _route(GEOCODE_OK, wet))
        assert "umbrella" in weather.get_weather("Delhi")

    def test_unknown_city_returns_empty(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get", _route({"results": []}, None))
        assert weather.get_weather("Nowhereville") == ""

    def test_missing_temperature_returns_empty(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get",
                            _route(GEOCODE_OK, {"current": {}}))
        assert weather.get_weather("Delhi") == ""

    def test_network_failure_returns_empty(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get",
                            _route(None, ConnectionError("down")))
        assert weather.get_weather("Delhi") == ""

    def test_geocoder_failure_returns_empty(self, _quiet, monkeypatch):
        monkeypatch.setattr(weather.requests, "get",
                            _route(ConnectionError("down"), FORECAST_OK))
        assert weather.get_weather("Delhi") == ""


class TestConditions:
    @pytest.mark.parametrize("code,expected", [
        (0, "clear"), (3, "overcast"), (95, "thunderstorm"), (999, "unsettled"),
    ])
    def test_wmo_codes(self, code, expected):
        assert weather._describe(code) == expected
