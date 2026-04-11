"""
chat_ui.py: for chatbot UI

Use : add to existing app.py:
    from src.app.chat_ui import render_chat_tab then call render_chat_tab() inside a st.tab() block
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import streamlit as st
from src.rag.rag_engine import ask, get_status
from src.rag.embeddings import build_full_index

# Index initialisation (runs once per session)
def _inject_chat_css():
    st.markdown("""
    <style>
    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        z-index: 999 !important;
        padding: 0.75rem 0 !important;
        background-color: #0e1117 !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: left 0.3s ease !important;
        /* NO width, NO right here — JS controls both left and right */
    }

    .stChatInput *,
    .stChatInput *:focus,
    .stChatInput *:active,
    .stChatInput *:focus-visible {
        box-shadow: none !important;
        outline: none !important;
        -webkit-box-shadow: none !important;
    }

    .stChatInput > div {
        border: 1px solid rgba(250, 100, 50, 0.5) !important;
        border-radius: 8px !important;
        background-color: #1a1f2e !important;
        position: relative !important;
        width: 100% !important;
    }

    .stChatInput > div:focus-within {
        border-color: rgba(255, 75, 75, 0.8) !important;
    }

    .stChatInput textarea {
        background-color: #262730 !important;
        border: none !important;
        color: #ffffff !important;
        padding: 0.75rem 3rem 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        resize: none !important;
        box-shadow: none !important;
        outline: none !important;
        width: 100% !important;
    }

    .stChatInput button {
        position: absolute !important;
        right: 0.25rem !important;
        bottom: 0.25rem !important;
        background-color: rgba(255,75,75,0) !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        padding: 0.5rem 0.5rem !important;    /* controls button size */
        min-width: unset !important;
        width: 2.8rem !important;               /* fixed width */
        height: 2.4rem !important;            /* fixed height */
    }
                
    /* Scale down the arrow icon inside */
                
    .stChatInput button svg {
        width: 18px !important;
        height: 18px !important;
    }
                
    .stChatInput button:hover {
        background-color: rgba(255,75,75, 0.9) !important;
        box-shadow: none !important;
        color: #fafafa !important;
        
    }

    .main .block-container {
        padding-bottom: 7rem !important;
    }
    </style>

    <script>
    const RIGHT_MARGIN = 24;   // px from right edge of viewport
    const LEFT_SIDEBAR = 336;  // px — 21rem sidebar width
    const LEFT_COLLAPSED = 28; // px — when sidebar is closed

    function adjustChatInput() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        const chatInput = document.querySelector('.stChatInput');
        if (!chatInput) return;

        const viewportWidth = window.innerWidth;
        const isOpen = sidebar
            ? sidebar.getAttribute('aria-expanded') === 'true'
            : false;

        const leftVal = isOpen ? LEFT_SIDEBAR : LEFT_COLLAPSED;
        const rightVal = RIGHT_MARGIN;

        // Set left, right AND width explicitly so nothing overflows
        chatInput.style.left = leftVal + 'px';
        chatInput.style.right = rightVal + 'px';
        chatInput.style.width = (viewportWidth - leftVal - rightVal) + 'px';
    }

    // Run on load with retries for Streamlit's async render
    adjustChatInput();
    setTimeout(adjustChatInput, 200);
    setTimeout(adjustChatInput, 600);

    // Watch sidebar toggle
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        new MutationObserver(adjustChatInput).observe(sidebar, {
            attributes: true,
            attributeFilter: ['aria-expanded']
        });
    }

    // Watch window resize
    window.addEventListener('resize', adjustChatInput);
    </script>
    """, unsafe_allow_html=True)

def _ensure_index_ready():
    """
    Checks if ChromaDB index exists.
    If not, triggers a build with a progress indicator.
    Shows a warning if index is empty.
    """
    status = get_status()

    if not status["ready"]:
        st.warning(
            "The search index is empty. Building it now — "
            "this takes a few minutes on first run..."
        )
        with st.spinner("Embedding complaints and recalls into ChromaDB..."):
            result = build_full_index()
        st.success(
            f"Index ready: {result['complaints']:,} complaints + "
            f"{result['recalls']:,} recalls indexed."
        )
        st.rerun()

    return status

# Main chat UI

def render_chat_tab():
    """
    Renders the full RAG chat interface.
    Call this inside your Streamlit app.
    """
    _inject_chat_css()
    st.header("Ask About Vehicle Safety")
    st.caption(
        "Ask any question about vehicle safety issues. "
        "Answers are grounded in real NHTSA complaints and recalls."
    )

    # Check index status
    status = _ensure_index_ready()

    # Show index stats in sidebar-style expander
    with st.expander("Index Status", expanded=True):
        col1, col2 = st.columns(2)
        col1.metric(
            "Total Complaints",
            f"{status.get('complaints_indexed', 0):,}"
        )
        col2.metric(
            "Recalls",
            f"{status.get('recalls_indexed', 0):,}"
        )

    st.divider()

    # Optional vehicle filter

    with st.expander("Filter by Vehicle (optional)", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            make_filter = st.text_input(
                "Make",
                placeholder="e.g. BMW",
                key="chat_make"
            )
        with filter_col2:
            model_filter = st.text_input(
                "Model",
                placeholder="e.g. 3 Series",
                key="chat_model"
            )
        with filter_col3:
            year_filter = st.number_input(
                "Year",
                min_value=1990,
                max_value=2026,
                value=None,
                placeholder="e.g. 2020",
                key="chat_year"
            )

    make = make_filter.strip() or None
    model = model_filter.strip() or None
    year = int(year_filter) if year_filter else None

    # Chat history

    # 1. Session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 2. Example questions
    if not st.session_state.chat_messages:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        example_questions = [
            "Are there engine fires in BMW 3 Series?",
            "What are the most common complaints for Toyota Camry 2019?",
            "Has the Ford F-150 been recalled for brake issues?",
            "Which vehicles have the most deaths reported?",
        ]
        for i, question in enumerate(example_questions):
            if cols[i % 2].button(question, key=f"example_{i}"):
                st.session_state.pending_question = question
                st.rerun()

    # 3. Display ALL messages including expanders, THIS is the only display loop
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message.get("badge"):
                st.caption(
                    f"{message['badge']}  •  {message.get('sources', '')}"
                )

            internal_docs = message.get("internal_docs", [])
            web_results = message.get("web_results", [])

            if internal_docs or web_results:
                total = len(internal_docs) + len(web_results)
                with st.expander(
                    f"View {total} sources", expanded=False
                ):
                    if internal_docs:
                        st.markdown("##### 🗂️ NHTSA Internal Data")
                        for i, doc in enumerate(internal_docs, 1):
                            score_pct = int(doc.score * 100)
                            icon = "🔴" if doc.doc_type == "recall" else "📋"
                            st.markdown(
                                f"**{icon} {doc.doc_type.upper()} {i} — "
                                f"{doc.make} {doc.model} {doc.year} — "
                                f"{score_pct}% match**"
                            )
                            st.text(
                                doc.text[:500] + "..."
                                if len(doc.text) > 500
                                else doc.text
                            )
                            if i < len(internal_docs):
                                st.divider()

                    if web_results:
                        if internal_docs:
                            st.divider()
                        st.markdown("##### 🌐 Web Sources")
                        for i, r in enumerate(web_results, 1):
                            st.markdown(f"**{i}. [{r.title}]({r.url})**")
                            st.text(
                                r.snippet[:500] + "..."
                                if len(r.snippet) > 500
                                else r.snippet
                            )
                            if i < len(web_results):
                                st.divider()

    # 4. Chat input
    typed_query = st.chat_input(
        "Ask about any vehicle safety issue...",
        key="chat_input"
    )

    # 5. Resolve query
    if "pending_question" in st.session_state:
        query = st.session_state.pop("pending_question")
    elif typed_query:
        query = typed_query
    else:
        query = None

    # 6. Process: save to session state then rerunn
    if query:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": query,
        })
        st.session_state.chat_history.append({
            "role": "user", "content": query,
        })

        with st.spinner("Searching complaints database and online resources..."):
            answer, sources, internal_docs, web_results, quality = ask(
                query=query,
                make=make,
                model=model,
                year=year,
                top_k=6,
                conversation_history=st.session_state.chat_history,
            )

        quality_badges = {
            "sufficient":   "🏛️ Official NHTSA Records",
            "partial":      "🏛️ NHTSA + 🌐 Web",
            "insufficient": "🌐 Web Search Only",
        }

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "badge": quality_badges.get(quality, ""),
            "internal_docs": internal_docs,
            "web_results": web_results,
        })
        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
        })
        st.rerun()

    # 7. Clear button
    if st.session_state.chat_messages:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_messages = []
            st.session_state.chat_history = []
            st.rerun()