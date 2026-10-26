"""Open-Meteo weather and geocoding client (no API key required).

* Live data: https://api.open-meteo.com/v1/forecast and
  https://geocoding-api.open-meteo.com/v1/search
* Every result is a :class:`WeatherSnapshot` that states where the numbers came
  from (``status`` = ``live`` | ``stale`` | ``simulated``).
* Live readings are never fabricated. If the API cannot be reached, callers can
  fall back to the last successful snapshot (marked ``stale``) or to a clearly
  labelled simulated diurnal profile (marked ``simulated``).
* This module has no Streamlit imports so it can be unit-tested on its own.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
USER_AGENT = "SafeRL-Building-Energy/2.0 (academic project)"
REQUEST_TIMEOUT = (5, 10)  # (connect, read) seconds
DEFAULT_TTL_SECONDS = 600  # Open-Meteo "current" data refreshes about every 15 minutes

CURRENT_VARS = ("temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
                "weather_code,wind_speed_10m,wind_direction_10m")
HOURLY_VARS = "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,shortwave_radiation"

STATUS_LIVE = "live"
STATUS_STALE = "stale"
STATUS_SIMULATED = "simulated"

SOURCE_LIVE = "Open-Meteo forecast API (live)"
SOURCE_SIMULATED = "Simulated sample weather (NOT a real reading)"

# Approximate city-centre coordinates for quick selection (no API call needed).
CITY_PRESETS: dict[str, tuple[float, float, str]] = {
    "Mumbai": (19.0760, 72.8777, "Asia/Kolkata"),
    "Navi Mumbai": (19.0330, 73.0297, "Asia/Kolkata"),
    "Thane": (19.2183, 72.9781, "Asia/Kolkata"),
    "Pune": (18.5204, 73.8567, "Asia/Kolkata"),
    "Delhi": (28.6139, 77.2090, "Asia/Kolkata"),
    "Bengaluru": (12.9716, 77.5946, "Asia/Kolkata"),
    "Hyderabad": (17.3850, 78.4867, "Asia/Kolkata"),
    "Chennai": (13.0827, 80.2707, "Asia/Kolkata"),
    "Kolkata": (22.5726, 88.3639, "Asia/Kolkata"),
    "Ahmedabad": (23.0225, 72.5714, "Asia/Kolkata"),
    "Jaipur": (26.9124, 75.7873, "Asia/Kolkata"),
}

# WMO weather interpretation codes (as documented by Open-Meteo).
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class WeatherError(Exception):
    """Raised for any failure to obtain or validate weather data."""


class LocationNotFound(WeatherError):
    """Raised when geocoding finds no match for the entered place."""


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""
    timezone: str = "auto"
    source: str = "preset"  # preset | geocoded

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


@dataclass
class WeatherSnapshot:
    location: Location
    status: str                      # live | stale | simulated
    source: str
    fetched_at: Optional[datetime]   # UTC time of the successful API call (None for simulated)
    current: dict = field(default_factory=dict)
    hourly: pd.DataFrame = field(default_factory=pd.DataFrame)
    warning: str = ""

    @property
    def is_live(self) -> bool:
        return self.status == STATUS_LIVE

    @property
    def description(self) -> Optional[str]:
        code = self.current.get("weather_code")
        if code is None:
            return None
        return WMO_CODES.get(int(code), f"WMO code {int(code)}")

    def age_minutes(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.fetched_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.fetched_at).total_seconds() / 60.0)


# ----------------------------------------------------------------------
# HTTP helper
# ----------------------------------------------------------------------
HttpGet = Callable[..., object]


def _default_get(url: str, params: dict, timeout=REQUEST_TIMEOUT):
    import requests  # local import keeps module import cheap and test-friendly
    return requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})


def _get_json(url: str, params: dict, http_get: Optional[HttpGet] = None) -> dict:
    http_get = http_get or _default_get
    try:
        response = http_get(url, params=params, timeout=REQUEST_TIMEOUT)
    except Exception as exc:  # timeouts, DNS, connection errors, ...
        raise WeatherError(f"Could not reach the weather service: {type(exc).__name__}: {exc}") from exc
    status = getattr(response, "status_code", 200)
    if status != 200:
        detail = ""
        try:
            detail = str(response.json().get("reason", ""))
        except Exception:
            pass
        raise WeatherError(f"Weather service returned HTTP {status}. {detail}".strip())
    try:
        payload = response.json()
    except Exception as exc:
        raise WeatherError("Weather service returned an unreadable response.") from exc
    if not isinstance(payload, dict):
        raise WeatherError("Weather service returned an unexpected response.")
    if payload.get("error"):
        raise WeatherError(f"Weather service error: {payload.get('reason', 'unknown reason')}")
    return payload


# ----------------------------------------------------------------------
# Location handling
# ----------------------------------------------------------------------
def preset_location(name: str) -> Location:
    key = next((k for k in CITY_PRESETS if k.lower() == name.strip().lower()), None)
    if key is None:
        raise LocationNotFound(f"'{name}' is not one of the preset cities.")
    lat, lon, tz = CITY_PRESETS[key]
    return Location(name=key, latitude=lat, longitude=lon, country="India", timezone=tz, source="preset")


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise WeatherError("Latitude and longitude must be numbers.") from exc
    if not (math.isfinite(lat) and math.isfinite(lon)) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise WeatherError("Latitude must be within -90..90 and longitude within -180..180.")
    return lat, lon


def geocode(query: str, count: int = 5, country_code: Optional[str] = None,
            http_get: Optional[HttpGet] = None) -> list[Location]:
    """Search Open-Meteo's geocoding API. Raises ``LocationNotFound`` for no match."""
    query = (query or "").strip()
    if len(query) < 2:
        raise LocationNotFound("Enter at least two characters for the location name.")
    params = {"name": query, "count": int(count), "language": "en", "format": "json"}
    if country_code:
        params["countryCode"] = country_code
    payload = _get_json(GEOCODING_URL, params, http_get)
    results = payload.get("results") or []
    locations: list[Location] = []
    for r in results:
        try:
            lat, lon = validate_coordinates(r["latitude"], r["longitude"])
        except (KeyError, WeatherError):
            continue
        locations.append(Location(
            name=str(r.get("name", query)), latitude=lat, longitude=lon,
            country=str(r.get("country", "")), admin1=str(r.get("admin1", "")),
            timezone=str(r.get("timezone", "auto")), source="geocoded"))
    if not locations:
        raise LocationNotFound(f"No location found for '{query}'. Check the spelling or try a larger nearby city.")
    return locations


# ----------------------------------------------------------------------
# Forecast parsing
# ----------------------------------------------------------------------
def _num(value, lo: float, hi: float) -> Optional[float]:
    """Return a float inside [lo, hi] or None (never substitutes a made-up value)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < lo or v > hi:
        return None
    return v


def parse_forecast(payload: dict, location: Location, fetched_at: Optional[datetime] = None) -> WeatherSnapshot:
    cur = payload.get("current")
    if not isinstance(cur, dict):
        raise WeatherError("Weather response did not include current conditions.")
    temperature = _num(cur.get("temperature_2m"), -60.0, 60.0)
    if temperature is None:
        raise WeatherError("Weather response is missing a valid current temperature.")
    current = {
        "time": cur.get("time"),
        "temperature_c": temperature,
        "humidity_pct": _num(cur.get("relative_humidity_2m"), 0.0, 100.0),
        "apparent_temperature_c": _num(cur.get("apparent_temperature"), -80.0, 70.0),
        "wind_speed_kmh": _num(cur.get("wind_speed_10m"), 0.0, 500.0),
        "wind_direction_deg": _num(cur.get("wind_direction_10m"), 0.0, 360.0),
        "weather_code": None if cur.get("weather_code") is None else int(cur["weather_code"]),
        "is_day": None if cur.get("is_day") is None else bool(cur["is_day"]),
    }

    hourly = pd.DataFrame()
    h = payload.get("hourly")
    if isinstance(h, dict) and h.get("time"):
        n = len(h["time"])
        frame = pd.DataFrame({"timestamp": pd.to_datetime(h["time"], errors="coerce")})
        for src, dst, lo, hi in (
            ("temperature_2m", "outdoor_temperature_c", -60.0, 60.0),
            ("relative_humidity_2m", "outdoor_humidity_pct", 0.0, 100.0),
            ("wind_speed_10m", "wind_speed_kmh", 0.0, 500.0),
            ("shortwave_radiation", "shortwave_radiation_wm2", 0.0, 1500.0),
            ("weather_code", "weather_code", 0.0, 120.0),
        ):
            series = h.get(src)
            if isinstance(series, list) and len(series) == n:
                frame[dst] = [_num(v, lo, hi) for v in series]
            else:
                frame[dst] = np.nan
        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        hourly = frame

    return WeatherSnapshot(
        location=location, status=STATUS_LIVE, source=SOURCE_LIVE,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        current=current, hourly=hourly,
    )


# ----------------------------------------------------------------------
# Cached service
# ----------------------------------------------------------------------
class WeatherService:
    """Fetches live weather with a TTL cache and explicit fallback behaviour."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, http_get: Optional[HttpGet] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = int(ttl_seconds)
        self._http_get = http_get
        self._clock = clock
        self._cache: dict[tuple[float, float], tuple[float, WeatherSnapshot]] = {}
        self._last_good: dict[tuple[float, float], WeatherSnapshot] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(location: Location) -> tuple[float, float]:
        return (round(location.latitude, 3), round(location.longitude, 3))

    def fetch_live(self, location: Location, force: bool = False, forecast_days: int = 3) -> WeatherSnapshot:
        """Return a live snapshot (cached for ``ttl_seconds`` unless ``force``). Raises ``WeatherError``."""
        key = self._key(location)
        with self._lock:
            cached = self._cache.get(key)
            if cached and not force and (self._clock() - cached[0]) < self.ttl_seconds:
                return cached[1]
        lat, lon = validate_coordinates(location.latitude, location.longitude)
        params = {
            "latitude": lat, "longitude": lon,
            "current": CURRENT_VARS, "hourly": HOURLY_VARS,
            "timezone": "auto", "forecast_days": int(forecast_days), "past_days": 1,
            "wind_speed_unit": "kmh", "temperature_unit": "celsius",
        }
        payload = _get_json(FORECAST_URL, params, self._http_get)
        snapshot = parse_forecast(payload, location)
        with self._lock:
            self._cache[key] = (self._clock(), snapshot)
            self._last_good[key] = snapshot
        return snapshot

    def get_weather(self, location: Location, force: bool = False, allow_simulated: bool = True
                    ) -> WeatherSnapshot:
        """Live weather, else last successful (stale) snapshot, else labelled simulation.

        The returned snapshot's ``status``/``source``/``warning`` always say which.
        """
        try:
            return self.fetch_live(location, force=force)
        except WeatherError as exc:
            error = str(exc)
        key = self._key(location)
        with self._lock:
            last = self._last_good.get(key)
        if last is not None:
            age = last.age_minutes()
            return WeatherSnapshot(
                location=last.location, status=STATUS_STALE,
                source=f"{SOURCE_LIVE} - last successful reading, now outdated",
                fetched_at=last.fetched_at, current=dict(last.current), hourly=last.hourly.copy(),
                warning=f"Live refresh failed ({error}). Showing the last successful reading "
                        f"from {age:.0f} min ago." if age is not None else f"Live refresh failed ({error}).",
            )
        if not allow_simulated:
            raise WeatherError(error)
        sim = simulated_weather(location)
        sim.warning = f"Live weather unavailable ({error}). Using simulated sample weather - NOT a real reading."
        return sim


# ----------------------------------------------------------------------
# Labelled simulated fallback
# ----------------------------------------------------------------------
def simulated_weather(location: Location, start: Optional[datetime] = None, hours: int = 72,
                      mean_temp_c: float = 30.0, swing_c: float = 5.0, mean_humidity: float = 60.0
                      ) -> WeatherSnapshot:
    """Smooth diurnal sample profile. Clearly labelled; never presented as live."""
    start = (start or datetime.now()).replace(minute=0, second=0, microsecond=0)
    times = pd.date_range(start - pd.Timedelta(hours=24), periods=hours + 24, freq="h")
    hour = times.hour.to_numpy(dtype=float)
    phase = np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    temp = mean_temp_c + swing_c * phase
    hum = np.clip(mean_humidity - 2.0 * swing_c * phase, 5.0, 100.0)
    radiation = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0)) * 800.0
    hourly = pd.DataFrame({
        "timestamp": times,
        "outdoor_temperature_c": temp, "outdoor_humidity_pct": hum,
        "wind_speed_kmh": np.nan, "shortwave_radiation_wm2": radiation,
        "weather_code": np.nan,
    })
    now_row = hourly.iloc[int(np.argmin(np.abs((hourly["timestamp"] - pd.Timestamp(start)).to_numpy()
                                               .astype("timedelta64[s]").astype(float))))]
    current = {
        "time": str(start), "temperature_c": float(now_row["outdoor_temperature_c"]),
        "humidity_pct": float(now_row["outdoor_humidity_pct"]), "apparent_temperature_c": None,
        "wind_speed_kmh": None, "wind_direction_deg": None, "weather_code": None, "is_day": None,
    }
    return WeatherSnapshot(location=location, status=STATUS_SIMULATED, source=SOURCE_SIMULATED,
                           fetched_at=None, current=current, hourly=hourly)
