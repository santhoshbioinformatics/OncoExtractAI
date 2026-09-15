"""Script to update app.py with MyDNA healthcare dashboard UI/UX design."""
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the CSS block and replace it
css_start = content.find('st.markdown(\n    """\n    <style>')
css_end = content.find('</style>\n    """', css_start) + len('</style>\n    """')

if css_start == -1 or css_end == -1:
    print("Could not find CSS section")
    exit(1)

# New CSS inspired by MyDNA healthcare dashboard
new_css = '''st.markdown(
    """
    <style>
    :root {
        --navy: #0b1f33; --teal: #0f6f78; --canvas: #f6f8fa;
        --paper: #fff; --text: #172b3a; --muted: #556b7c;
        --line: #dbe3ea; --soft: #edf2f6; --radius: 12px;
        --green: #246b45; --green-bg: #e5f4ea;
        --amber: #7a4b00; --amber-bg: #fff1d6;
        --red: #8f2929; --red-bg: #fbe7e7;
        --accent: #4a90e2; --accent-light: #e8f4fd;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--canvas); }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
    
    /* Sidebar - Dark navy with icon navigation */
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #f2f7fa; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        min-height: 44px; border-radius: 10px; padding: .4rem .6rem;
        margin: 2px 0;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stCaption p { color: inherit; }
    
    /* Profile card in sidebar */
    [data-testid="stSidebar"] .sidebar-profile {
        background: rgba(255,255,255,.14) !important;
        border: 1px solid rgba(255,255,255,.28) !important;
        border-radius: 12px !important;
        margin: .35rem 0 .75rem !important;
        padding: .85rem .9rem !important;
    }
    [data-testid="stSidebar"] .sidebar-profile,
    [data-testid="stSidebar"] .sidebar-profile * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .sidebar-profile-name {
        font-size: 1rem !important;
        font-weight: 750 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .sidebar-profile-status {
        color: #c5e4ec !important;
        font-size: .8rem !important;
        margin: .25rem 0 0 !important;
        opacity: 1 !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] div[data-testid="stButton"] button,
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"],
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #ffffff !important;
        color: #0b1f33 !important;
        font-weight: 750 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #d9eef2 !important;
        border-color: #d9eef2 !important;
        color: #0b1f33 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button *,
    [data-testid="stSidebar"] div[data-testid="stButton"] button p,
    [data-testid="stSidebar"] div[data-testid="stButton"] button span {
        color: #0b1f33 !important;
    }
    
    /* Typography */
    h1, h2, h3 { color: var(--navy); letter-spacing: -.02em; }
    h1 { font-size: clamp(1.75rem, 3vw, 2.25rem); }
    h2 { font-size: clamp(1.3rem, 2vw, 1.6rem); }
    p, label, li, button, input, textarea, select { font-size: .94rem; }
    small, .stCaption p { font-size: .78rem !important; }
    :focus-visible { outline: 3px solid #18a3ad !important; outline-offset: 2px !important; }
    a { color: #0b6570; } a:hover { color: #084b53; }
    
    /* Page headings */
    .page-heading { margin-bottom: 1rem; }
    .page-heading h1 { margin: 0 0 .35rem; line-height: 1.15; }
    .page-heading p { color: var(--muted); margin: 0; max-width: 880px; line-height: 1.55; }
    .section-heading { color: var(--navy); font-size: 1rem; font-weight: 750; margin: 0 0 .55rem; }
    
    /* Research banner */
    .research-banner {
        background: #fff8e8; border: 1px solid #ead29b; border-left: 4px solid #a56b05;
        border-radius: var(--radius); color: #624308; font-size: .86rem;
        line-height: 1.45; margin: 0 0 1rem; padding: .7rem .85rem;
    }
    
    /* Stats cards - MyDNA style */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        transition: box-shadow 0.2s;
    }
    .stat-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    .stat-icon-blue { background: var(--accent-light); }
    .stat-icon-green { background: var(--green-bg); }
    .stat-icon-amber { background: var(--amber-bg); }
    .stat-icon-red { background: var(--red-bg); }
    .stat-content { flex: 1; }
    .stat-label { color: var(--muted); font-size: .82rem; font-weight: 600; margin: 0 0 .25rem; }
    .stat-value { color: var(--navy); font-size: 1.75rem; font-weight: 800; margin: 0; line-height: 1.1; }
    .stat-trend { font-size: .78rem; font-weight: 600; margin: .25rem 0 0; }
    .stat-trend-up { color: var(--green); }
    .stat-trend-down { color: var(--red); }
    
    /* Case strip */
    .case-strip, .priority-strip {
        background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius);
        margin: .75rem 0 1rem; padding: .8rem .95rem;
    }
    .case-strip { align-items: center; display: flex; flex-wrap: wrap; gap: .55rem 1rem; }
    .case-id { color: var(--navy); font-weight: 750; overflow-wrap: anywhere; }
    .case-meta { color: var(--muted); font-size: .8rem; overflow-wrap: anywhere; }
    .priority-strip p { color: #40586b; font-size: .84rem; line-height: 1.5; margin: .35rem 0 0; }
    
    /* Status pills */
    .status-pill, .priority-pill {
        align-items: center; border: 1px solid transparent; border-radius: 999px;
        display: inline-flex; font-size: .75rem; font-weight: 750; line-height: 1.2;
        max-width: 100%; min-height: 26px; padding: .28rem .58rem; white-space: normal;
    }
    .tone-positive { background: var(--green-bg); color: var(--green); border-color: #bfdfca; }
    .tone-neutral { background: var(--soft); color: #405465; border-color: #d7e0e6; }
    .tone-warning { background: var(--amber-bg); color: var(--amber); border-color: #ecd69e; }
    .tone-danger { background: var(--red-bg); color: var(--red); border-color: #edc4c4; }
    
    /* Field cards */
    .field-card, .review-card, .method-panel {
        background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius);
        margin-bottom: .75rem; padding: .9rem 1rem;
    }
    .field-top { align-items: flex-start; display: flex; flex-wrap: wrap;
        gap: .6rem 1rem; justify-content: space-between; }
    .field-label, .comparison-label { color: var(--muted); font-size: .74rem;
        font-weight: 750; letter-spacing: .055em; text-transform: uppercase; }
    .field-value { color: var(--navy); font-size: 1.02rem; font-weight: 720;
        line-height: 1.4; margin-top: .25rem; overflow-wrap: anywhere; }
    .field-note, .distinction-note { color: var(--muted); font-size: .8rem;
        line-height: 1.45; margin-top: .5rem; overflow-wrap: anywhere; }
    .distinction-note { background: #eaf3fb; border-radius: 7px; color: #254f70; padding: .5rem .6rem; }
    
    /* Evidence navigation */
    .anchor-link { display: inline-block; font-size: .8rem; font-weight: 650; margin-top: .55rem; }
    .evidence-nav { align-items: center; display: flex; flex-wrap: wrap;
        gap: .45rem; margin: .4rem 0 .75rem; }
    .evidence-nav a { background: var(--paper); border: 1px solid var(--line);
        border-radius: 7px; font-size: .78rem; font-weight: 650;
        padding: .35rem .55rem; text-decoration: none; }
    
    /* Report reader */
    .report-reader { background: var(--paper); border: 1px solid var(--line);
        border-radius: var(--radius); color: #203746;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .84rem; line-height: 1.7; max-height: 68vh; min-height: 360px;
        overflow: auto; overflow-wrap: anywhere; padding: 1rem 1.05rem; white-space: pre-wrap; }
    .evidence-anchor { scroll-margin-top: 1rem; }
    .evidence-mark { background: #d9f1ea; border-bottom: 2px solid #3c9b7c;
        border-radius: 2px; color: #183e32; padding: 1px 0; }
    .evidence-mark--focused { background: #ffe7a8; border-bottom-color: #af7200;
        box-shadow: 0 0 0 2px rgba(175,114,0,.18); }
    .evidence-excerpt { background: #f8fafb; border: 1px solid var(--line);
        border-left: 3px solid #55959b; border-radius: 7px; color: #294150;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .82rem; line-height: 1.55; margin: .4rem 0 .75rem;
        overflow-wrap: anywhere; padding: .65rem .75rem; white-space: pre-wrap; }
    .offset-label { color: var(--muted); font-size: .76rem; font-weight: 650; }
    
    /* Empty panels */
    .empty-panel { background: var(--paper); border: 1px dashed #b8c7d2;
        border-radius: var(--radius); color: var(--muted); line-height: 1.5; padding: 1.25rem; }
    .review-card h3 { font-size: 1rem; margin: 0 0 .15rem; }
    .method-panel { min-height: 150px; }
    .method-panel--evidence { border-top: 3px solid var(--teal); }
    .method-name { color: var(--navy); font-weight: 750; }
    .method-value { font-size: .98rem; font-weight: 680; margin: .55rem 0; overflow-wrap: anywhere; }
    
    /* Status legend */
    .status-legend { display: grid; gap: .55rem; grid-template-columns: repeat(2,minmax(0,1fr)); }
    .legend-item { background: var(--paper); border: 1px solid var(--line);
        border-radius: var(--radius); padding: .75rem; }
    .legend-item p { color: var(--muted); font-size: .8rem; line-height: 1.4; margin: .4rem 0 0; }
    
    /* Streamlit component overrides */
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
    [data-testid="stExpander"] { background: var(--paper); border-color: var(--line); border-radius: var(--radius); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; }
    [data-testid="stTabs"] [data-baseweb="tab"] { min-height: 44px; white-space: nowrap; }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
        border-radius: 8px; min-height: 40px; font-weight: 650;
    }
    #MainMenu, footer { visibility: hidden; }
    
    /* Welcome greeting */
    .welcome-greeting {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--navy);
        margin: 0 0 .25rem;
    }
    .welcome-subtitle {
        color: var(--muted);
        font-size: .95rem;
        margin: 0 0 1rem;
    }
    
    @media (max-width: 820px) {
        .block-container { padding: 1rem .75rem 2rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="column"] { flex: 1 1 280px !important;
            min-width: min(100%,280px) !important; width: 100% !important; }
        .report-reader { max-height: 55vh; min-height: 300px; }
        .status-legend { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)'''

content = content[:css_start] + new_css + content[css_end:]

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("CSS updated successfully!")
