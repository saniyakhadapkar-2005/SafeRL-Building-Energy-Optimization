from datetime import datetime, timezone

import pandas as pd
import pytest

from utils import weather as W

LOC = W.preset_location("Mumbai")


def payload(temp=31.2, hum=64, code=1):
    times = pd.date_range("2026-10-25 00:00", periods=96, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "current": {"time": "2026-10-25T10:00", "temperature_2m": temp, "relative_humidity_2m": hum,
                    "apparent_temperature": 35.0, "wind_speed_10m": 9.5, "wind_direction_10m": 200,
                    "weather_code": code, "is_day": 1},
        "hourly": {"time": times, "temperature_2m": [30.0] * 96, "relative_humidity_2m": [60] * 96,
                   "wind_speed_10m": [8] * 96, "shortwave_radiation": [0] * 96, "weather_code": [1] * 96},
    }


class Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def json(self):
        if isinstance(self._d, Exception):
            raise self._d
        return self._d


def getter(response):
    calls = []

    def _get(url, params=None, timeout=None):
        calls.append((url, params))
        if isinstance(response, Exception):
            raise response
        return response
    _get.calls = calls
    return _get


def test_presets_cover_required_cities():
    for c in ("Mumbai", "Pune", "Delhi", "Bengaluru"):
        assert W.preset_location(c).latitude
    with pytest.raises(W.LocationNotFound):
        W.preset_location("Atlantis")


def test_parse_live_snapshot():
    svc = W.WeatherService(http_get=getter(Resp(payload())))
    snap = svc.get_weather(LOC)
    assert snap.status == W.STATUS_LIVE and snap.is_live
    assert snap.current["temperature_c"] == 31.2 and snap.current["humidity_pct"] == 64
    assert snap.description == "Mainly clear"
    assert len(snap.hourly) == 96 and snap.fetched_at is not None


def test_timeout_falls_back_to_labelled_simulation():
    svc = W.WeatherService(http_get=getter(TimeoutError("timed out")))
    snap = svc.get_weather(LOC)
    assert snap.status == W.STATUS_SIMULATED
    assert "NOT a real reading" in snap.source and "unavailable" in snap.warning
    assert snap.fetched_at is None


def test_no_fallback_option_raises():
    svc = W.WeatherService(http_get=getter(TimeoutError("x")))
    with pytest.raises(W.WeatherError):
        svc.get_weather(LOC, allow_simulated=False)


def test_http_error_and_bad_json():
    with pytest.raises(W.WeatherError):
        W.WeatherService(http_get=getter(Resp({}, status=500))).fetch_live(LOC)
    with pytest.raises(W.WeatherError):
        W.WeatherService(http_get=getter(Resp(ValueError("bad")))).fetch_live(LOC)
    with pytest.raises(W.WeatherError):
        W.WeatherService(http_get=getter(Resp({"error": True, "reason": "limit"}))).fetch_live(LOC)


def test_missing_or_invalid_values():
    with pytest.raises(W.WeatherError):
        W.parse_forecast({"current": {"time": "x"}}, LOC)
    with pytest.raises(W.WeatherError):
        W.parse_forecast({"current": {"temperature_2m": 999}}, LOC)
    with pytest.raises(W.WeatherError):
        W.parse_forecast({}, LOC)
    snap = W.parse_forecast(payload(hum=150), LOC)   # humidity out of range -> None, not invented
    assert snap.current["humidity_pct"] is None


def test_cache_and_force_refresh():
    t = [0.0]
    g = getter(Resp(payload()))
    svc = W.WeatherService(ttl_seconds=600, http_get=g, clock=lambda: t[0])
    svc.fetch_live(LOC)
    svc.fetch_live(LOC)
    assert len(g.calls) == 1                 # cached
    svc.fetch_live(LOC, force=True)
    assert len(g.calls) == 2
    t[0] = 700
    svc.fetch_live(LOC)
    assert len(g.calls) == 3                 # TTL expired


def test_stale_snapshot_used_when_refresh_fails():
    t = [0.0]
    good = getter(Resp(payload()))
    svc = W.WeatherService(ttl_seconds=1, http_get=good, clock=lambda: t[0])
    svc.get_weather(LOC)
    svc._http_get = getter(TimeoutError("down"))
    t[0] = 100
    snap = svc.get_weather(LOC)
    assert snap.status == W.STATUS_STALE and "outdated" in snap.source and snap.current["temperature_c"] == 31.2


def test_geocode_results_and_not_found():
    data = {"results": [{"name": "Nagpur", "latitude": 21.1458, "longitude": 79.0882, "country": "India",
                         "admin1": "Maharashtra", "timezone": "Asia/Kolkata"}]}
    locs = W.geocode("Nagpur", http_get=getter(Resp(data)))
    assert locs[0].label == "Nagpur, Maharashtra, India" and locs[0].source == "geocoded"
    with pytest.raises(W.LocationNotFound):
        W.geocode("zzzzqq", http_get=getter(Resp({})))
    with pytest.raises(W.LocationNotFound):
        W.geocode("a")


def test_coordinate_validation():
    assert W.validate_coordinates(19, 72) == (19.0, 72.0)
    for bad in ((91, 0), (0, 181), ("x", 1), (float("nan"), 1)):
        with pytest.raises(W.WeatherError):
            W.validate_coordinates(*bad)
