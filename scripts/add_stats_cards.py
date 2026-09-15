"""Add stats cards and welcome greeting to the workspace page."""

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the page_workspace function and replace it with enhanced version
old_workspace = '''def page_workspace() -> None:
    _page_heading(
        "Report Workspace",
        "Load approved text, run one extraction method, inspect exact evidence, and record reviewer decisions.",
    )
    _research_banner()
    _render_source_loader()'''

new_workspace = '''def page_workspace() -> None:
    _page_heading(
        "Report Workspace",
        "Load approved text, run one extraction method, inspect exact evidence, and record reviewer decisions.",
    )
    _research_banner()
    
    # Stats cards
    reports_loaded = len(st.session_state.get("loaded_reports", {}))
    extractions_run = len(st.session_state.get("extraction_results", {}))
    reviews_saved = sum(1 for k in st.session_state.get("review_decisions", {}).keys() if st.session_state["review_decisions"][k])
    audit_records = len(st.session_state.get("audit_records", []))
    
    st.markdown(
        f'<div class="stats-grid">'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-blue">📄</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Reports Loaded</p>'
        f'<p class="stat-value">{reports_loaded}</p>'
        f'</div></div>'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-green">⚙️</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Extractions Run</p>'
        f'<p class="stat-value">{extractions_run}</p>'
        f'</div></div>'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-amber">✅</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Reviews Saved</p>'
        f'<p class="stat-value">{reviews_saved}</p>'
        f'</div></div>'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-red">📊</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Audit Records</p>'
        f'<p class="stat-value">{audit_records}</p>'
        f'</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    _render_source_loader()'''

content = content.replace(old_workspace, new_workspace)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Stats cards added successfully!")
