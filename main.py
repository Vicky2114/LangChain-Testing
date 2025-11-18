from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os
import dotenv
from websockets.exceptions import ConnectionClosedError
import json
import pprint
import re
import uuid
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional
import time
from contextlib import asynccontextmanager

dotenv.load_dotenv()

# Load API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=api_key)

DEFAULT_POSTGRES_URL = (
    os.getenv("POSTGRES_URL")
    or f"postgresql://{os.getenv('USER', 'bigstep')}@localhost:5432/langchain_memory"
)
if not DEFAULT_POSTGRES_URL:
    raise ValueError(
        "POSTGRES_URL environment variable is not set and no default could be derived."
    )

# Initialize Gemini model
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # You can also use "gemini-1.5-flash" for faster responses
    google_api_key=api_key,
    temperature=0.2,
    convert_system_message_to_human=True
)

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle (light)",
    57: "Freezing drizzle (dense)",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain (light)",
    67: "Freezing rain (heavy)",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Rain showers (slight)",
    81: "Rain showers (moderate)",
    82: "Rain showers (violent)",
    85: "Snow showers (slight)",
    86: "Snow showers (heavy)",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

WEATHER_CACHE_TTL_SECONDS = 300
_weather_cache: Dict[str, Dict[str, Any]] = {}
agent: Optional[Any] = None


class WeatherServiceError(Exception):
    """Raised when the upstream weather API cannot fulfill a request."""


def describe_weather_code(code: int) -> str:
    return WEATHER_CODE_MAP.get(code, "Weather data unavailable")


def geocode_city(city: str) -> Dict[str, Any]:
    if not city:
        raise WeatherServiceError("Please provide a city name.")
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        raise WeatherServiceError(f"Could not find a match for '{city}'.")
    result = results[0]
    return {
        "name": result.get("name"),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
        "country": result.get("country"),
        "admin1": result.get("admin1"),
        "timezone": result.get("timezone"),
    }


def fetch_weather_bundle(city: str) -> Dict[str, Any]:
    normalized_city = city.strip().lower()
    cached = _weather_cache.get(normalized_city)
    if cached and time.time() - cached["timestamp"] < WEATHER_CACHE_TTL_SECONDS:
        return cached["payload"]

    location = geocode_city(city)
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_mean,weather_code",
        "forecast_days": 3,
        "timezone": "auto",
    }
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast", params=params, timeout=10
    )
    response.raise_for_status()
    data = response.json()

    print("data --------------------------", data)

    current = data.get("current", {})
    hourly_data = data.get("hourly", {})
    daily_data = data.get("daily", {})

    hourly = []
    times = hourly_data.get("time", [])
    temps = hourly_data.get("temperature_2m", [])
    precip = hourly_data.get("precipitation_probability", [])
    codes = hourly_data.get("weather_code", [])
    for idx in range(min(6, len(times))):
        hourly.append(
            {
                "time": datetime.fromisoformat(times[idx]).strftime("%I:%M %p"),
                "temperature": temps[idx],
                "precipitation_probability": precip[idx],
                "summary": describe_weather_code(codes[idx]),
            }
        )

    daily = []
    daily_times = daily_data.get("time", [])
    max_temps = daily_data.get("temperature_2m_max", [])
    min_temps = daily_data.get("temperature_2m_min", [])
    daily_precip = daily_data.get("precipitation_probability_mean", [])
    daily_codes = daily_data.get("weather_code", daily_precip)
    for idx in range(min(3, len(daily_times))):
        daily.append(
            {
                "day": datetime.fromisoformat(daily_times[idx]).strftime("%A"),
                "high": max_temps[idx],
                "low": min_temps[idx],
                "precipitation_probability": daily_precip[idx],
                "summary": describe_weather_code(daily_codes[idx]),
            }
        )

    current_precip = hourly[0]["precipitation_probability"] if hourly else 0

    payload = {
        "location": {
            "city": location["name"],
            "region": location.get("admin1"),
            "country": location.get("country"),
            "timezone": location.get("timezone"),
        },
        "current": {
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "summary": describe_weather_code(current.get("weather_code", -1)),
            "weather_code": current.get("weather_code"),
            "precipitation_probability": current_precip,
            "as_of": current.get("time"),
        },
        "hourly": hourly,
        "daily": daily,
    }

    _weather_cache[normalized_city] = {"timestamp": time.time(), "payload": payload}
    return payload


def _format_location_label(payload: Dict[str, Any]) -> str:
    location = payload.get("location", {})
    pieces = [location.get("city")]
    if location.get("region"):
        pieces.append(location["region"])
    if location.get("country"):
        pieces.append(location["country"])
    return ", ".join([part for part in pieces if part])


def build_weather_blocks(city: str, payload: Dict[str, Any]) -> str:
    location_label = _format_location_label(payload)
    current = payload["current"]
    hourly_lines = [
        f"- {entry['time']}: {entry['temperature']}°C, {entry['summary']} (rain {entry['precipitation_probability']}%)"
        for entry in payload["hourly"]
    ]
    daily_lines = [
        f"- {entry['day']}: {entry['summary']}, {entry['low']}°C / {entry['high']}°C, rain {entry['precipitation_probability']}%"
        for entry in payload["daily"]
    ]

    planner_tip = "Carry a light layer and stay hydrated."
    if current.get("temperature") is not None:
        temp = current["temperature"]
        if temp >= 30:
            planner_tip = "Plan for heat: hydrate, wear sunscreen, and limit midday sun."
        elif temp <= 5:
            planner_tip = "Bundle up and watch for icy spots during commutes."
    if current.get("wind_speed", 0) > 35:
        planner_tip = "Strong winds expected—secure outdoor items and expect a brisk feel."

    return (
        f"[Current Conditions]\n"
        f"- Location: {location_label}\n"
        f"- Temp: {current.get('temperature')}°C (feels {current.get('apparent_temperature')}°C)\n"
        f"- Humidity: {current.get('humidity')}%\n"
        f"- Wind: {current.get('wind_speed')} km/h\n"
        f"- Skies: {current.get('summary')}\n\n"
        f"[Next 6 Hours]\n" + "\n".join(hourly_lines) + "\n\n"
        f"[Planner Tips]\n"
        f"- {planner_tip}\n"
        f"- Upcoming days:\n" + "\n".join(daily_lines)
    )


def get_current_weather(city: str) -> str:
    """Get a block-formatted snapshot for a city."""
    payload = fetch_weather_bundle(city)
    return build_weather_blocks(city, payload)


def get_hourly_forecast(city: str) -> str:
    """Return a compact hourly look ahead."""
    payload = fetch_weather_bundle(city)
    lines = [
        f"{entry['time']}: {entry['temperature']}°C, {entry['summary']} (rain {entry['precipitation_probability']}%)"
        for entry in payload["hourly"]
    ]
    location_label = _format_location_label(payload)
    return "[Hourly Outlook]\n" + "\n".join(lines) + f"\n\nSource: {location_label}"


def get_outdoor_planner(city: str) -> str:
    """Provide practical guidance for making plans."""
    payload = fetch_weather_bundle(city)
    current = payload["current"]
    advice = []
    if current.get("precipitation_probability", 0) and current["precipitation_probability"] > 50:
        advice.append("Pack rain gear for the next few hours.")
    if current.get("wind_speed", 0) > 30:
        advice.append("Windy periods could impact cycling or rooftop plans.")
    if not advice:
        advice.append("Weather looks friendly—perfect for outdoor errands or a walk.")
    advice.append("Check the 3-day outlook for temperature swings.")
    return "[Planner]\n- " + "\n- ".join(advice)

def build_agent(checkpointer):
    return create_agent(
    model=gemini_model,
        tools=[get_current_weather, get_hourly_forecast, get_outdoor_planner],
        system_prompt=(
            "You are a helpful assistant powered by Google Gemini. Be concise and friendly. "
            "For weather-related answers, respond using clearly labeled blocks: "
            "[Current Conditions], [Next 6 Hours], and [Planner Tips]. "
            "Reference available tools whenever structured data is needed."
        ),
        checkpointer=checkpointer,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    cm = AsyncPostgresSaver.from_conn_string(conn_string=DEFAULT_POSTGRES_URL)
    checkpointer = await cm.__aenter__()
    await checkpointer.setup()
    agent = build_agent(checkpointer)
    try:
        yield
    finally:
        agent = None
        await cm.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/weather")
async def api_weather(city: str):
    try:
        payload = fetch_weather_bundle(city)
        return payload
    except WeatherServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Upstream weather service error") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to fetch weather right now.") from exc

async def safe_send_text(ws: WebSocket, message: str) -> bool:
    """Safely send text through WebSocket, returns False if connection is closed."""
    try:
        await ws.send_text(message)
        return True
    except (WebSocketDisconnect, ConnectionClosedError, ConnectionError):
        return False
    except Exception as e:
        print(f"Error sending message: {e}")
        return False

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Generate a unique thread_id for this WebSocket session
    # This allows the agent to remember conversation history across messages
    thread_id = str(uuid.uuid4())
    print(f"New WebSocket connection with thread_id: {thread_id}")
    if agent is None:
        await ws.close(code=1013, reason="Agent initializing, please reconnect.")
        print("Agent not ready; closed connection.")
        return
    
    try:
        await ws.accept()
        # Try to send welcome message, but don't fail if connection is already closed
        await safe_send_text(ws, "🤖 WebSocket Connected! Ask me anything...")
    except Exception as e:
        print(f"Error accepting WebSocket: {e}")
        return

    try:
        while True:
            try:
                user_input = await ws.receive_text()
            except WebSocketDisconnect:
                print(f"Client disconnected (thread_id: {thread_id})")
                break

            # Prepare input for agent (LangChain v1.0 format)
            # Only send the user message - agent will retrieve full history from checkpointer
            agent_input = {
                "messages": [HumanMessage(content=user_input)]
            }
            
            # Configure agent to use this thread's memory
            config = {
                "configurable": {
                    "thread_id": thread_id  # Use same thread_id to maintain conversation history
                }
            }

            # Stream response using LangChain's astream_events
            # Reference: https://docs.langchain.com/oss/python/langchain/models
            # This provides real-time token streaming from the agent
            # The config ensures conversation history is maintained via thread_id
            final_message = ""
            has_sent_content = False
            try:
                print(f"Processing user input (thread_id: {thread_id}): {user_input}")
                async for event in agent.astream_events(agent_input, config=config, version="v2"):
                    # Check if connection is still open before processing
                    if ws.client_state.name != "CONNECTED":
                        break
                    
                    event_type = event.get("event")
                    event_name = event.get("name", "")
                    event_data = event.get("data", {})

                    # Emit detailed event debug logs for observability
                    print(f"[Agent Event] type={event_type} name={event_name}")
                    if event_type in {"on_tool_start", "on_tool_end"}:
                        tool_name = (
                            event_data.get("tool", {}).get("name")
                            if isinstance(event_data.get("tool"), dict)
                            else event_name
                        )
                        print(f"  ↳ Tool invocation: {tool_name}")
                        if "input" in event_data:
                            print(f"    input: {event_data['input']}")
                        if "output" in event_data:
                            print(f"    output: {event_data['output']}")
                    
                    # Optional: Debug output (comment out for production)
                    # print(f"Event: {event_type} | Name: {event_name}")
                    
                    # Look for chat model streaming events - these are the token chunks
                    if event_type == "on_chat_model_stream":
                        chunk = event_data.get("chunk")
                        if chunk:
                            content = None
                            # Handle chunk as a message object (has content attribute)
                            if hasattr(chunk, "content"):
                                chunk_content = chunk.content
                                if isinstance(chunk_content, str):
                                    content = chunk_content
                                elif isinstance(chunk_content, list):
                                    # Extract text from list of dicts with type: 'text'
                                    text_parts = []
                                    for item in chunk_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            text_parts.append(item.get("text", ""))
                                    content = "".join(text_parts) if text_parts else None
                            # Handle chunk as string representation (parse it)
                            elif isinstance(chunk, str):
                                # Parse string like "content='Hello! How can' additional_kwargs={}..."
                                # Handle escaped quotes and newlines
                                match = re.search(r"content='((?:[^'\\]|\\.)*)'", chunk)
                                if match:
                                    content = match.group(1).replace("\\n", "\n").replace("\\'", "'")
                            # Handle chunk as dict
                            elif isinstance(chunk, dict):
                                if "content" in chunk:
                                    chunk_content = chunk["content"]
                                    if isinstance(chunk_content, str):
                                        content = chunk_content
                                    elif isinstance(chunk_content, list):
                                        text_parts = []
                                        for item in chunk_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_parts.append(item.get("text", ""))
                                        content = "".join(text_parts) if text_parts else None
                                elif "text" in chunk:
                                    content = str(chunk["text"])
                            
                            # Only send non-empty content (skip empty chunks and last empty chunk)
                            if content and content.strip():
                                final_message += content
                                has_sent_content = True
                                if not await safe_send_text(ws, content):
                                    break  # Connection closed, exit loop
                    
                    # Catch chain streaming events (usually contains messages)
                    elif event_type == "on_chain_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            # Extract messages from chunk
                            if isinstance(chunk, dict) and "messages" in chunk:
                                messages = chunk["messages"]
                                if messages:
                                    # Get the last message (usually the AI response)
                                    last_msg = messages[-1]
                                    if hasattr(last_msg, "content"):
                                        content = last_msg.content
                                        if isinstance(content, str) and content.strip():
                                            if content not in final_message:
                                                final_message += content
                                                has_sent_content = True
                                                if not await safe_send_text(ws, content):
                                                    break
                                    elif isinstance(last_msg, str):
                                        # Parse string representation
                                        match = re.search(r"content='((?:[^'\\]|\\.)*)'", last_msg)
                                        if match:
                                            content = match.group(1).replace("\\n", "\n").replace("\\'", "'")
                                            if content.strip() and content not in final_message:
                                                final_message += content
                                                has_sent_content = True
                                                if not await safe_send_text(ws, content):
                                                    break
                    
                    # Catch final output when agent completes (only if we haven't streamed yet)
                    elif event_type == "on_chain_end":
                        # Only process LangGraph final output if we haven't streamed content
                        if event_name == "LangGraph" and not has_sent_content:
                            output = event.get("data", {}).get("output")
                            if output and isinstance(output, dict) and "messages" in output:
                                messages = output["messages"]
                                if messages:
                                    # Get the last message (AI response)
                                    last_msg = messages[-1]
                                    if hasattr(last_msg, "content"):
                                        content = last_msg.content
                                        if isinstance(content, str) and content.strip():
                                            if content not in final_message:
                                                final_message += content
                                                has_sent_content = True
                                                if not await safe_send_text(ws, content):
                                                    break
                                    elif isinstance(last_msg, str):
                                        # Parse string representation
                                        match = re.search(r"content='((?:[^'\\]|\\.)*)'", last_msg)
                                        if match:
                                            content = match.group(1).replace("\\n", "\n").replace("\\'", "'")
                                            if content.strip() and content not in final_message:
                                                final_message += content
                                                has_sent_content = True
                                                if not await safe_send_text(ws, content):
                                                    break
                
                # If no content was streamed, log it (shouldn't happen with proper streaming)
                if not has_sent_content:
                    print("⚠️ Warning: No content was streamed from agent events")
                    await safe_send_text(ws, "⚠️ No response received. Please try again.")
                        
            except (WebSocketDisconnect, ConnectionClosedError, ConnectionError):
                print("Connection closed during streaming")
                break
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                print(f"Agent error: {e}")
                import traceback
                traceback.print_exc()
                await safe_send_text(ws, error_msg)
                # Don't break, allow user to try again

    except WebSocketDisconnect:
        print("Client disconnected")
    except ConnectionClosedError:
        print("Connection closed")
    except Exception as e:
        print(f"Unexpected error: {e}")
        try:
            await safe_send_text(ws, f"⚠️ Error: {str(e)}")
        except:
            pass
    finally:
        try:
            await ws.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
