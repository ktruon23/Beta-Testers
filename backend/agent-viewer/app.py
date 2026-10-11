"""
Beta Agents live viewer.

Shows which stage the system is in (detecting, planning, critiquing, coaching) and what
each agent is thinking, live, as the Planner and Critic negotiate a route.

Start the agent services first (Vision 8001, Planner 8002, Critic 8003, Coach 8004), then:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io

import httpx
import streamlit as st
from PIL import Image, ImageDraw

import pipeline

st.set_page_config(page_title="Beta Agents", layout="wide")

AGENTS = {  # stage -> (display name, avatar)
    "detect": ("Vision", "👁️"),
    "filter": ("Orchestrator", "🧭"),
    "holds": ("Orchestrator", "🧭"),
    "plan": ("Planner", "🗺️"),
    "critique": ("Critic", "🧐"),
    "grade": ("Critic", "🧐"),
    "coach": ("Coach", "🧗"),
    "error": ("System", "⚠️"),
}
STAGE_LABEL = {
    "detect": "Detecting holds",
    "filter": "Picking the route color",
    "plan": "Planning a route",
    "critique": "Critiquing the route",
    "grade": "Grading the route",
    "coach": "Writing the beta",
}


def draw(image_bytes: bytes, holds: list[dict], path: list[int] | None = None,
         avoid: set[int] | None = None) -> Image.Image:
    """Holds as small circles, the route as a numbered line, avoided holds in red."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    d = ImageDraw.Draw(img)
    r = max(3, img.width // 120)
    by_id = {h["id"]: h for h in holds}
    for h in holds:
        color = (255, 60, 60) if avoid and h["id"] in avoid else (255, 255, 255)
        d.ellipse([h["x"] - r, h["y"] - r, h["x"] + r, h["y"] + r], outline=color, width=2)
        d.text((h["x"] + r + 2, h["y"] - r), str(h["id"]), fill=color)  # id, for setting hold types
    if path:
        pts = [(by_id[i]["x"], by_id[i]["y"]) for i in path if i in by_id]
        d.line(pts, fill=(0, 230, 120), width=max(2, r // 2))
        for n, (x, y) in enumerate(pts):
            d.ellipse([x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6], fill=(0, 230, 120))
            d.text((x - r / 2, y - r), str(n), fill=(0, 0, 0))
    return img


# ---------------------------------------------------------------- sidebar: inputs

with st.sidebar:
    st.header("Climber")
    height = st.number_input("Height (in)", 48.0, 90.0, 68.0)
    wingspan = st.number_input("Wingspan (in)", 48.0, 95.0, height)
    level = st.selectbox("Level", ["beginner", "intermediate", "expert"], index=1)
    reach_factor = st.slider("Reach factor", 0.6, 1.1, 0.85, 0.01,
                             help="Longest single move = wingspan x reach factor")
    st.caption(f"Longest allowed move: {wingspan * reach_factor:.0f} in")

    st.header("Holds")
    default_hold = st.selectbox("Default hold type", pipeline.HOLD_TYPES,
                                index=pipeline.HOLD_TYPES.index(pipeline.DEFAULT_HOLD_TYPE),
                                help="Used for every hold the camera can't identify")
    hold_types_text = st.text_input("Known hold types", placeholder="3=crimp, 5=sloper",
                                    help="Hold id = type, using the ids drawn on the photo")
    try:
        hold_types = pipeline.parse_hold_types(hold_types_text)
    except ValueError as e:
        st.error(str(e))
        hold_types = {}

    st.header("Agents")
    max_rounds = st.slider("Max Planner/Critic rounds", 1, 8, 4)
    min_conf = st.slider("Min detection confidence", 0.1, 0.9, 0.5, 0.05)
    if st.button("Check agents"):
        for name, url in [("Vision", pipeline.VISION_URL), ("Planner", pipeline.PLANNER_URL),
                          ("Critic", pipeline.CRITIC_URL), ("Coach", pipeline.COACH_URL)]:
            try:
                info = httpx.get(url + "/", timeout=5).json()
                llm = " · Claude" if info.get("llm") else ""
                st.write(f"🟢 {name}{llm}")
            except httpx.HTTPError:
                st.write(f"🔴 {name} ({url})")

climber = {"height_in": height, "wingspan_in": wingspan, "reach_factor": reach_factor, "level": level}

# ---------------------------------------------------------------- step 1: photo + detection

st.title("Beta Agents")
upload = st.file_uploader("Wall photo", type=["jpg", "jpeg", "png"])
if upload is None:
    st.info("Upload a photo of the wall to start.")
    st.stop()

image_bytes = upload.getvalue()
if st.session_state.get("image_name") != upload.name:
    st.session_state.update(image_name=upload.name, detection=None, result=None)

left, right = st.columns([1, 1])

if st.session_state.detection is None:
    with right:
        with st.status("Vision is detecting holds...", expanded=True) as s:
            try:
                det = httpx.post(f"{pipeline.VISION_URL}/detect", timeout=pipeline.TIMEOUT, json={
                    "image_b64": base64.b64encode(image_bytes).decode(), "min_conf": min_conf}).json()
                st.session_state.detection = det
                s.update(label=f"Vision found {len(det['holds'])} holds", state="complete", expanded=False)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                s.update(label="Vision failed", state="error")
                st.error(f"Could not reach the Vision agent at {pipeline.VISION_URL}: {e}")
                st.stop()

det = st.session_state.detection
with right:
    for line in pipeline.explain_detection(det):
        st.write(line)
    colors = list(det.get("colors", {}))
    route_color = st.selectbox(
        "Route color", ["(all holds)", *colors],
        format_func=lambda c: c if c == "(all holds)" else f"{c} ({det['colors'][c]} holds)")
    route_color = None if route_color == "(all holds)" else route_color
    go = st.button("Find my route", type="primary")

with left:
    image_slot = st.empty()
    shown = [h for h in det["holds"] if route_color is None or h.get("color") == route_color]
    image_slot.image(draw(image_bytes, shown), caption="Detected holds", use_container_width=True)

# ---------------------------------------------------------------- step 2: live agent run

if go:
    st.session_state.result = None
    with right:
        st.subheader("What the agents are doing")
        stage_box = st.empty()
        feed = st.container()

    holds_for_route = shown
    avoid: set[int] = set()
    status_widget = None
    for ev in pipeline.run(image_bytes, climber, route_color=route_color, min_conf=min_conf,
                           max_rounds=max_rounds, detection=det,
                           hold_types=hold_types, default_hold_type=default_hold):
        name, avatar = AGENTS.get(ev.stage, ("System", "ℹ️"))

        if ev.stage in STAGE_LABEL and ev.status == "running":
            # Live "working" indicator; the agent's thinking is written into it when the step finishes.
            stage_box.info(f"**Now:** {STAGE_LABEL[ev.stage]}" + (f" (round {ev.round})" if ev.round else ""))
            with feed:
                status_widget = st.status(f"{avatar} {ev.title}", expanded=True)
            continue

        if ev.stage in ("filter", "holds"):
            holds_for_route = ev.data["holds"]
        if ev.stage == "critique" and not ev.data.get("approved", True):
            avoid |= set(ev.data.get("avoid_holds", []))
        if ev.stage == "plan" and ev.data.get("ok"):
            image_slot.image(draw(image_bytes, holds_for_route, ev.data["path"], avoid),
                             caption=f"Round {ev.round}: Planner's proposal", use_container_width=True)

        if ev.stage == "done":
            st.session_state.result = ev.data
            stage_box.success(f"**Done:** {ev.title}")
            continue

        if status_widget is not None:
            with status_widget:
                for line in ev.detail:
                    st.markdown(f"- {line}")
            status_widget.update(label=f"{avatar} {ev.title}", expanded=True,
                                 state="error" if ev.status == "failed" else "complete")
            status_widget = None
        else:
            with feed:
                with st.chat_message(name, avatar=avatar):
                    st.markdown(f"**{name}**: {ev.title}")
                    for line in ev.detail:
                        st.markdown(f"- {line}")
        if ev.stage == "error":
            stage_box.error(ev.title)

# ---------------------------------------------------------------- final route

res = st.session_state.get("result")
if res and res.get("plan"):
    with left:
        image_slot.image(draw(image_bytes, res["holds"], res["plan"]["path"]),
                         caption=("Final route (approved by the Critic)" if res["approved"]
                                  else "Final route (not approved)"),
                         use_container_width=True)
        diff = res.get("difficulty")
        if diff:
            st.metric("Difficulty for you", diff["grade"], diff["label"], delta_color="off")
            for line in pipeline.explain_difficulty(diff)[1:]:
                st.caption(line)
        st.markdown("#### Moves (calculated, not guessed)")
        st.dataframe([{
            "move": f"#{m['from_id']} → #{m['to_id']}",
            "distance (in)": m["distance_in"],
            "reach": f"{m['reach_ratio']:.0%}",
            "up (in)": m["dy_in"],
            "right (in)": m["dx_in"],
            "to hold": m["to_type"],
        } for m in res["plan"]["moves"]], hide_index=True, use_container_width=True)
