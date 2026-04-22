import streamlit as st
import httpx
import os

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="MeetingMind",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 MeetingMind")
    st.caption("Your team's AI memory")
    st.divider()

    workspace_id = st.text_input(
        "Workspace ID",
        value="my-team",
        help="Each team has its own workspace"
    )

    st.divider()
    st.subheader("Add a Meeting")

    source_type = st.selectbox(
        "Source type",
        ["zoom", "google-meet", "teams", "slack", "email", "pdf"]
    )
    title = st.text_input("Title", placeholder="Q2 Planning Meeting")
    transcript = st.text_area(
        "Paste transcript here",
        height=200,
        placeholder="[00:00:05] Alice: Let's discuss the roadmap..."
    )

    if st.button("Ingest", type="primary", use_container_width=True):
        if transcript.strip() and workspace_id:
            with st.spinner("Ingesting..."):
                try:
                    import uuid
                    r = httpx.post(f"{API_URL}/ingest", json={
                        "workspace_id": workspace_id,
                        "source_id": str(uuid.uuid4())[:8],
                        "text": transcript,
                        "source_type": source_type,
                        "title": title or "Untitled",
                    }, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        st.success(
                            f"Done! {data['chunks_created']} chunks indexed."
                        )
                    else:
                        st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")
        else:
            st.warning("Paste a transcript first.")

# ── Main chat ─────────────────────────────────────────────────
st.title("Ask your meetings anything")
st.caption(f"Workspace: `{workspace_id}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! Paste a meeting transcript in the sidebar, then ask me anything about it.",
        "sources": [],
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    st.caption(
                        f"**{s['source_id']}** — "
                        f"score: {s['score']} — "
                        f"\"{s['excerpt'][:100]}...\""
                    )

if prompt := st.chat_input("What did we decide about the launch date?"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": [],
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching your meetings..."):
            try:
                r = httpx.post(f"{API_URL}/query", json={
                    "workspace_id": workspace_id,
                    "question": prompt,
                    "top_k": 5,
                }, timeout=60)

                if r.status_code == 200:
                    data = r.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 {len(sources)} source(s)"):
                            for s in sources:
                                st.caption(
                                    f"**{s['source_id']}** — "
                                    f"score: {s['score']} — "
                                    f"\"{s['excerpt'][:100]}...\""
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error(f"Backend error: {r.status_code}")

            except httpx.ConnectError:
                st.error("Cannot reach backend. Make sure FastAPI is running.")