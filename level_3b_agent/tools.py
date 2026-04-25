"""Tools for level_3b_agent.

Mixed real / mock:
  - `get_weather`     → weather_checker. **Real**, calls the public
                        Open-Meteo API (no key, no auth). Geocodes the
                        city, then fetches current conditions + a
                        3-day forecast.
  - `search_flights`  → flight_booker. **Mocked**. Google does not
                        publish a public flights API; see notes below.
  - `book_flight`     → flight_booker. **Mocked** — never books a real
                        flight, returns a deterministic confirmation.

Why mixed?
----------
Open-Meteo is free, requires no key, and is stable — there's no reason
to mock it. Live data also makes the demo more compelling.

Flights are mocked because there is no public Google Flights API.
Google's old QPX Express API was sunset in 2018. The current options
are:
  - **Duffel** — real booking-grade API, sandbox tier free, requires
    OAuth (~10 min to set up).
  - **Amadeus Self-Service** — free tier (2000 calls/month), requires
    API key + secret, real booking sandbox.
  - **SerpAPI** Google Flights endpoint — paid scraper, $50+/month
    after a small free tier; gives you the *Google Flights* listing
    specifically.
  - **Skyscanner Travel API** — partner-only.

For this demo, mocked flights keep the agent's lesson (`mode='task'`
HITL clarification) front and center without dragging in API auth.
"""

from __future__ import annotations

import httpx


# ---------------------------------------------------------------------------
# weather_checker tool — real Open-Meteo data, no API key required.
# ---------------------------------------------------------------------------


# WMO weather interpretation codes → human strings. Open-Meteo returns
# these on `weather_code`. Reference: https://open-meteo.com/en/docs
# (search "WMO Weather interpretation codes"). Only the most common
# buckets are mapped — anything not listed falls through to a generic
# label so the agent never sees a bare integer.
_WMO_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Light freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with light hail",
    99: "Thunderstorm with heavy hail",
}


def _describe_wmo(code: int | None) -> str:
  if code is None:
    return "Unknown"
  return _WMO_CODE_DESCRIPTIONS.get(int(code), f"Weather code {code}")


def get_weather(city: str) -> dict:
  """Return current conditions and a 3-day outlook for a city.

  Real implementation backed by Open-Meteo's free public API
  (https://open-meteo.com — no API key required). Two HTTP calls:
  geocoding to resolve `city` → lat/lon, then forecast.

  Args:
    city: A city name in any reasonable form ("Paris", "San Francisco",
      "Tokyo"). The geocoder picks the highest-population match;
      ambiguous names ("Springfield") may resolve unexpectedly — the
      returned dict echoes the resolved location so the agent can flag
      that to the user if needed.

  Returns:
    A dict with:
      - `city`: echo of the input
      - `resolved_location`: "City, Region, Country" from the geocoder
        (lets the agent confirm the right place was found)
      - `current`: {temp_c, conditions, humidity}
      - `outlook`: list of up to 3 day-by-day entries
        ({date, conditions, temp_c_max, temp_c_min})

    On any error (geocoding miss, network failure), returns a dict
    with an `error` key explaining the failure — the agent should
    surface that to the user rather than fabricating weather.
  """
  geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
  forecast_url = "https://api.open-meteo.com/v1/forecast"

  with httpx.Client(timeout=10.0) as client:
    try:
      geo_resp = client.get(
          geocode_url,
          params={"name": city, "count": 1, "language": "en", "format": "json"},
      )
      geo_resp.raise_for_status()
      geo_data = geo_resp.json()
      results = geo_data.get("results") or []
      if not results:
        return {
            "city": city,
            "error": (
                f"Could not find a location matching '{city}'. Ask the"
                " user to clarify (e.g., add a country)."
            ),
        }
      hit = results[0]
      latitude = hit["latitude"]
      longitude = hit["longitude"]
      resolved_parts = [
          hit.get("name"),
          hit.get("admin1"),
          hit.get("country"),
      ]
      resolved_location = ", ".join(p for p in resolved_parts if p)

      forecast_resp = client.get(
          forecast_url,
          params={
              "latitude": latitude,
              "longitude": longitude,
              "current": "temperature_2m,relative_humidity_2m,weather_code",
              "daily": "temperature_2m_max,temperature_2m_min,weather_code",
              "forecast_days": 4,  # today + 3 outlook days
              "timezone": "auto",
          },
      )
      forecast_resp.raise_for_status()
      f_data = forecast_resp.json()
    except httpx.HTTPError as exc:
      return {
          "city": city,
          "error": (
              f"Open-Meteo request failed: {exc!r}. The user should be"
              " told the lookup failed; do not invent weather."
          ),
      }

  current = f_data.get("current", {})
  daily = f_data.get("daily", {})
  daily_dates = daily.get("time", [])
  daily_codes = daily.get("weather_code", [])
  daily_max = daily.get("temperature_2m_max", [])
  daily_min = daily.get("temperature_2m_min", [])

  # `current` includes today; the outlook is the next 3 days. Skip
  # index 0 (today) so the outlook is purely forward-looking.
  outlook = []
  for idx in range(1, min(4, len(daily_dates))):
    outlook.append({
        "date": daily_dates[idx],
        "conditions": _describe_wmo(
            daily_codes[idx] if idx < len(daily_codes) else None
        ),
        "temp_c_max": (
            daily_max[idx] if idx < len(daily_max) else None
        ),
        "temp_c_min": (
            daily_min[idx] if idx < len(daily_min) else None
        ),
    })

  return {
      "city": city,
      "resolved_location": resolved_location,
      "current": {
          "temp_c": current.get("temperature_2m"),
          "conditions": _describe_wmo(current.get("weather_code")),
          "humidity": current.get("relative_humidity_2m"),
      },
      "outlook": outlook,
  }


# ---------------------------------------------------------------------------
# flight_booker tools
# ---------------------------------------------------------------------------


def search_flights(origin: str, destination: str, date: str) -> list[dict]:
  """Return up to 3 mock flight options for the route on the given date.

  Mocked: stable, deterministic results. The price_usd, stops, and
  airline distribution are spread so the LLM has a real choice to
  surface to the user.

  Args:
    origin: IATA airport code or city name (e.g. "SFO" or "San Francisco").
    destination: IATA airport code or city name (e.g. "CDG" or "Paris").
    date: Departure date in ISO format (YYYY-MM-DD). Required — the
      booker MUST ask the user for this; the coordinator is forbidden
      from guessing one. (See agent.py instruction.)

  Returns:
    A list of dicts, each with `flight_id`, `airline`, `depart_iso`,
    `arrive_iso`, `price_usd`, `stops`.
  """
  return [
      {
          "flight_id": "AA42",
          "airline": "American Airlines",
          "depart_iso": f"{date}T08:30",
          "arrive_iso": f"{date}T22:15",
          "price_usd": 845,
          "stops": 0,
      },
      {
          "flight_id": "DL128",
          "airline": "Delta",
          "depart_iso": f"{date}T11:00",
          "arrive_iso": f"{date}T23:50",
          "price_usd": 712,
          "stops": 1,
      },
      {
          "flight_id": "AF83",
          "airline": "Air France",
          "depart_iso": f"{date}T14:20",
          "arrive_iso": f"{date}T22:45",
          "price_usd": 925,
          "stops": 0,
      },
  ]


def book_flight(flight_id: str) -> dict:
  """Reserve the given flight and return a confirmation.

  Mocked: returns a fixed PNR for any flight_id. A real implementation
  would call an airline reservation system and validate seat
  availability + payment.

  Args:
    flight_id: The `flight_id` from one of the options returned by
      `search_flights`.

  Returns:
    A dict with `flight_id`, `pnr`, and `status`.
  """
  return {"flight_id": flight_id, "pnr": "ABCXYZ", "status": "CONFIRMED"}
