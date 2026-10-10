"""Weather tool for JARVIS 2.0.

Uses Open-Meteo: no API key, no signup, no scraping. The previous Google
scrape fallback is gone -- it parsed brittle CSS classes from a page that
serves different markup depending on who is asking, so it silently returned
nothing. Open-Meteo needs a geocoding lookup to turn a city name into
coordinates, which it also provides for free.

Returns data or an honest failure. Never invents a temperature.
"""

import requests

from jarvis.config import DEFAULT_CITY, WEATHER_TIMEOUT_S, USER_AGENT
from jarvis.logger import logger

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: Open-Meteo WMO weather interpretation codes -> plain English. Only the
#: codes that actually occur are listed; anything else falls through to
#: "conditions" rather than being reported as clear sky.
_WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}

#: WMO codes that imply precipitation, used for the umbrellas line.
_WET = set(range(51, 68)) | set(range(80, 88)) | {95, 96, 99}


def _describe(code) -> str:
    """Plain-English conditions for a WMO code."""
    try:
        return _WMO.get(int(code), "unsettled")
    except (TypeError, ValueError):
        return "unsettled"


def _geocode(city: str):
    """(latitude, longitude, resolved name) for a city, or None."""
    try:
        response = requests.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "format": "json"},
            timeout=WEATHER_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        results = (response.json() or {}).get("results") or []
        if not results:
            logger.debug(f"Open-Meteo geocoding found no match for {city!r}")
            return None
        best = results[0]
        return float(best["latitude"]), float(best["longitude"]), best.get("name", city)
    except Exception as e:  # noqa: BLE001 -- reported as "no data", never fatal
        logger.debug(f"Open-Meteo geocoding failed for {city!r}: {e}")
        return None


def _forecast(lat: float, lon: float):
    """Current conditions dict, or None. Empty/malformed responses mean None."""
    try:
        response = requests.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=WEATHER_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        data = response.json() or {}
        current = data.get("current") or {}
        temp = current.get("temperature_2m")
        if temp is None:
            logger.debug(f"Open-Meteo returned no current temperature (status={response.status_code})")
            return None

        daily = data.get("daily") or {}
        result = {
            "temperature": round(float(temp)),
            "feels_like": (
                round(float(current["apparent_temperature"]))
                if current.get("apparent_temperature") is not None else None
            ),
            "humidity": current.get("relative_humidity_2m"),
            "code": current.get("weather_code"),
            "high": _first(daily.get("temperature_2m_max")),
            "low": _first(daily.get("temperature_2m_min")),
            "rain_chance": _first(daily.get("precipitation_probability_max")),
        }
        return result
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Open-Meteo forecast failed at {lat},{lon}: {e}")
        return None


def _first(values):
    """First element of a WMO single-day list, or None."""
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return values if values is not None and not isinstance(values, (list, tuple)) else None


def get_weather(city: str = None) -> str:
    """Current weather for a city as a spoken-ready sentence.

    Args:
        city: City name. Falls back to the configured default city.

    Returns:
        A one-line description, or "" when no source had data.
    """
    target_city = city or DEFAULT_CITY
    place = _geocode(target_city)
    if place is None:
        logger.warning(f"Weather unavailable: could not locate {target_city!r}")
        return ""
    lat, lon, resolved = place

    data = _forecast(lat, lon)
    if data is None:
        logger.warning(f"Weather unavailable: no current conditions for {resolved!r}")
        return ""

    line = f"{resolved}: {data['temperature']} degrees Celsius"
    if data["high"] is not None and data["low"] is not None:
        line += f", {data['low']} to {data['high']} today"
    conditions = _describe(data.get("code")) if data.get("code") is not None else ""
    if conditions:
        line += f", {conditions}"
    if data.get("feels_like") is not None and data["feels_like"] != data["temperature"]:
        line += f", feels like {data['feels_like']} degrees"
    if (data.get("rain_chance") or 0) >= 40 or (
        data.get("code") is not None and int(data["code"]) in _WET
    ):
        line += ", you will probably want an umbrella"

    logger.info(f"Weather for {resolved!r}: {line}")
    return line
