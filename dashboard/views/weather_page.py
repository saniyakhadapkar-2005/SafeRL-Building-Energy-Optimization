"""Weather and Location page."""
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import charts, components, state, ui
from utils.weather import (CITY_PRESETS, STATUS_LIVE, Location, LocationNotFound, WeatherError, geocode,
                           preset_location, validate_coordinates)

CUSTOM = "Custom location…"


def _local(dt) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if dt else "—"


def render() -> None:
    components.header()
    ui.flash_messages()
    s = state.settings()
    st.subheader("Location")
    options = list(CITY_PRESETS) + [CUSTOM]
    idx = options.index(s["location_choice"]) if s["location_choice"] in options else 0
    choice = st.selectbox("City", options, index=idx)
    if choice != CUSTOM:
        if choice != s["location_choice"] or s["location"].source != "preset":
            s["location"] = preset_location(choice)
        s["location_choice"] = choice
    else:
        s["location_choice"] = CUSTOM
        mode = st.radio("Enter location by", ["Search by name", "Latitude / longitude"], horizontal=True)
        if mode == "Search by name":
            q = st.text_input("Place name (e.g. Nagpur, Jaipur, Kochi)", value="")
            if ui.button("Search", key="geo_search") and q:
                try:
                    st.session_state.geo_results = geocode(q, count=5)
                except WeatherError as exc:
                    st.session_state.geo_results = []
                    st.error(str(exc))
            results = st.session_state.get("geo_results") or []
            if results:
                pick = st.selectbox("Matches", range(len(results)), format_func=lambda i: results[i].label)
                if ui.button("Use this location", key="geo_use"):
                    s["location"] = results[pick]
                    st.success(f"Selected {results[pick].label}")
        else:
            c1, c2, c3 = st.columns(3)
            lat = c1.number_input("Latitude", -90.0, 90.0, float(s["location"].latitude), 0.01, format="%.4f")
            lon = c2.number_input("Longitude", -180.0, 180.0, float(s["location"].longitude), 0.01, format="%.4f")
            name = c3.text_input("Label", value="Custom point")
            if ui.button("Use these coordinates", key="coord_use"):
                try:
                    la, lo = validate_coordinates(lat, lon)
                    s["location"] = Location(name=name or "Custom point", latitude=la, longitude=lo, source="manual")
                except WeatherError as exc:
                    st.error(str(exc))
    loc = state.location()
    st.markdown(f"**Selected:** {loc.label} · latitude **{loc.latitude:.4f}**, longitude **{loc.longitude:.4f}** · timezone {loc.timezone}")

    st.subheader("Current outdoor weather")
    c1, c2 = st.columns([1, 3])
    c1.button("🌦 Refresh weather", key="wx_refresh", on_click=state.cb_refresh_weather, **ui._stretch(st.button))
    snap = state.ensure_weather()
    c2.markdown(components.weather_badge(snap), unsafe_allow_html=True)
    s["weather_ttl_min"] = int(st.number_input("Minimum minutes between live API calls", 1, 120, int(s["weather_ttl_min"]),
                                               help="Repeated refreshes inside this window reuse the cached reading (use the button to force)."))

    if snap.warning:
        st.warning(snap.warning)
    cur = snap.current
    m = st.columns(5)
    m[0].metric("Temperature", ui.fmt(cur.get("temperature_c"), ".1f", " °C"))
    m[1].metric("Humidity", ui.fmt(cur.get("humidity_pct"), ".0f", " %"))
    m[2].metric("Wind", ui.fmt(cur.get("wind_speed_kmh"), ".1f", " km/h"))
    m[3].metric("Conditions", snap.description or "—")
    m[4].metric("Feels like", ui.fmt(cur.get("apparent_temperature_c"), ".1f", " °C"))
    st.markdown(
        f"- **Data source:** {snap.source}\n"
        f"- **Status:** {snap.status.upper()}" + (" — real API reading" if snap.is_live else " — NOT a current real reading") + "\n"
        f"- **Last successful live update:** {_local(snap.fetched_at)}"
        + (f" ({snap.age_minutes():.0f} min ago)" if snap.fetched_at else " (none for this view)") + "\n"
        f"- **Observation time reported by API:** {cur.get('time') or '—'}")

    hourly = snap.hourly
    if hourly is not None and len(hourly):
        now = pd.to_datetime(cur.get("time"), errors="coerce")
        window = hourly[(hourly["timestamp"] >= (now - pd.Timedelta(hours=12) if pd.notna(now) else hourly["timestamp"].min()))
                        & (hourly["timestamp"] <= (now + pd.Timedelta(hours=48) if pd.notna(now) else hourly["timestamp"].max()))]
        label = "Open-Meteo hourly forecast" if snap.is_live or snap.status == "stale" else "SIMULATED sample profile"
        ui.show_chart(charts.weather_forecast_chart(window, now, label), "wx_chart")
    st.caption("Outdoor weather is only an input to the simulation. Indoor temperatures are produced by the simulated "
               "building model, never copied from outdoor readings.")
