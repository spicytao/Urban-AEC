import streamlit as st
import plotly.graph_objects as go
import requests
import os
import math
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 0. Load Environment & Architect's Monochrome CSS
# ==========================================
load_dotenv()

# Set page layout to wide mode
st.set_page_config(layout="wide", page_title="URBAN GENERATIVE AGENT", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* 彻底解决白色 Bar 的问题：将背景强制设为纯黑，完美隐身 */
    [data-testid="stHeader"] { background-color: #000000 !important; }
    /* 隐藏右上角没用的 Deploy 等菜单栏 */
    [data-testid="stToolbar"] { display: none !important; }
    
    /* Pure black background with high-contrast text */
    .stApp { background-color: #000000; color: #F0F0F0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .block-container { padding-top: 2rem !important; } 
    
    h1, h2, h3, h4, h5 { font-weight: 500 !important; color: #FFFFFF; letter-spacing: 0.05em; text-transform: uppercase; }
    h1 { font-size: 1.8rem !important; border-bottom: 1px solid #333333; padding-bottom: 15px; margin-bottom: 1rem; }
    
    /* Buttons: Minimalist wireframe style */
    .stButton > button { 
        background-color: transparent !important; color: #FFFFFF !important; 
        border: 1px solid #FFFFFF !important; border-radius: 0px !important; 
        font-weight: 400; letter-spacing: 1px; padding: 10px 24px; width: 100%; transition: 0.3s; 
    }
    .stButton > button:hover { background-color: #FFFFFF !important; color: #000000 !important; }
    
    /* Input fields */
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #1A1A1A; }
    .stTextInput>div>div>input, .stTextArea textarea, .stNumberInput input { 
        border-radius: 0px !important; background-color: #111111 !important; 
        border: 1px solid #444444 !important; color: #FFFFFF !important; 
    }
    
    hr { border-top: 1px solid #333333; margin: 1.5rem 0; }
    
    /* Metric Cards: 提高边框和内部文字的亮度 */
    .metric-card { background: transparent; border: 1px solid #444444; padding: 15px; margin-bottom: 15px; border-left: 2px solid #777777; }
    .metric-title { font-size: 0.75rem; color: #BBBBBB; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; font-weight: bold; }
    .metric-value { font-size: 1.6rem; color: #FFFFFF; font-weight: 300; line-height: 1.2;}
    
    /* Evaluation Badges: 提亮边框和文字 */
    .eval-badge { border: 1px solid #777; color: #DDDDDD; padding: 3px 8px; font-size: 0.75rem; display: inline-block; margin-bottom: 8px; font-family: monospace;}
    .eval-good { border-color: #FFF; color: #FFF; font-weight: bold; }
    .eval-warn { border-color: #BBBBBB; color: #BBBBBB; }
    .eval-bad { border: 1px solid #FF5555; background: #220000; color: #FF5555; font-weight: bold; } 
    
    /* 提亮占位符和描述文本 */
    .placeholder-box { border: 1px dashed #555; text-align: center; padding: 40px 20px; color: #999999; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;}
    .rationale-text { font-size: 0.9rem; color: #DDDDDD; line-height: 1.6; border-left: 2px solid #666; padding-left: 15px; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

def get_api_key(key_name):
    try: return st.secrets[key_name]
    except Exception: return os.getenv(key_name)

# ==========================================
# 1. Geocoding & OSM Extraction Engine
# ==========================================
def geocode_address(address):
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
    headers = {'User-Agent': 'AEC_Research_Agent/1.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        if res: return float(res[0]['lat']), float(res[0]['lon']), res[0]['display_name']
        return None, None, "Location not found."
    except Exception as e: return None, None, str(e)

def latlon_to_meters(lat, lon, clat, clon):
    r = 6378137.0; dx = (lon - clon) * (math.pi/180.0) * r * math.cos(clat * math.pi/180.0); dy = (lat - clat) * (math.pi/180.0) * r
    return dx, dy

def fetch_urban_context(clat, clon, r=150):
    query = f"[out:json];(way[\"building\"](around:{r},{clat},{clon}););out body; >; out skel qt;"
    try:
        data = requests.post("http://overpass-api.de/api/interpreter", data={'data': query}, timeout=15).json()
        nodes = {el['id']: latlon_to_meters(el['lat'], el['lon'], clat, clon) for el in data.get('elements', []) if el['type'] == 'node'}
        buildings = []
        for el in data.get('elements', []):
            if el['type'] == 'way' and 'tags' in el and 'building' in el['tags']:
                h = float(el['tags'].get('height', '15').replace('m','').strip()) if 'height' in el['tags'] else float(el['tags'].get('building:levels', 4)) * 3.5
                fp = [nodes[n] for n in el.get('nodes', []) if n in nodes]
                if len(fp) > 2: buildings.append({"id": el['id'], "footprint": fp, "height": h})
        return {"buildings": buildings}
    except Exception as e: return {"error": str(e)}

# ==========================================
# 2. Agent 1: Lead Architect
# ==========================================
class BuildingBlock(BaseModel):
    width: float = Field(description="Width along X-axis (m)"); length: float = Field(description="Length along Y-axis (m)"); height: float = Field(description="Height (m)")
    offset_x: float = Field(description="X-axis offset from center (m)"); offset_y: float = Field(description="Y-axis offset from center (m)"); elevation: float = Field(description="Elevation from ground (m)")

class MassingProposal(BaseModel):
    rationale: str = Field(description="A professional architectural statement explaining the design logic. Must address site context, massing strategy, and program distribution.")
    blocks: list[BuildingBlock] = Field(description="A composition of 1 to 4 parametric blocks.")

def generate_massing(api_key, context_str, intent):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o", temperature=0.6) 
    structured_llm = llm.with_structured_output(MassingProposal)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a world-class Lead Architect specializing in urban design. Your task is to generate a conceptual massing proposal using 1 to 4 parametric blocks.
        
        【PROFESSIONAL DIRECTIVES】:
        1. **Spatial Syntax & Program**: Do not just stack boxes. Consider creating courtyards for light penetration, cantilevers for covered public plazas, or setbacks for terraces.
        2. **Contextual Response**: If the site density is high, prioritize setbacks to reduce street-level oppression. If a public square is needed, consider lifting the ground floor (elevation > 0 for some blocks).
        3. **Geometric Constraints**: The first block must be grounded (elevation=0). Stacked blocks must align precisely on the Z-axis with the block below them.
        
        【TONE OF VOICE】: Your rationale should sound like a design statement from a top-tier firm (e.g., OMA, BIG, Foster + Partners), demonstrating a deep understanding of urban fabric, scale, and programmatic clarity.
        """), 
        ("human", f"SITE CONTEXT: {context_str}\nCLIENT INTENT: {intent}")
    ])
    return (prompt | structured_llm).invoke({})

# ==========================================
# 3. Agent 2: Urban Impact Assessor
# ==========================================
def evaluate_urban_impact(blocks, context_buildings):
    total_volume = 0
    max_h = 0
    max_w = 0; max_l = 0
    cx_sum = 0; cy_sum = 0
    
    for b in blocks:
        total_volume += b.width * b.length * b.height
        max_h = max(max_h, b.elevation + b.height)
        max_w = max(max_w, b.width); max_l = max(max_l, b.length)
        cx_sum += b.offset_x; cy_sum += b.offset_y
    
    center_x = cx_sum / len(blocks) if blocks else 0
    center_y = cy_sum / len(blocks) if blocks else 0
    
    impact_radius = max_h * 1.5
    impacted_neighbors = []
    
    for bldg in context_buildings:
        pts = bldg['footprint']
        bx = sum(p[0] for p in pts) / len(pts)
        by = sum(p[1] for p in pts) / len(pts)
        dist = math.sqrt((bx - center_x)**2 + (by - center_y)**2)
        
        if dist < impact_radius and max_h > bldg['height'] * 0.8:
            bldg['impacted'] = True
            impacted_neighbors.append(bldg)
        else:
            bldg['impacted'] = False
            
    report = {}
    
    footprint_area = max_w * max_l
    if footprint_area > 3000:
        report['massing'] = {"score": "[-] OVER-SCALED", "desc": "Excessive ground floor footprint disrupts street grain. Consider breaking down the mass or lifting for permeability."}
    else:
        report['massing'] = {"score": "[+] HUMAN-SCALE", "desc": "Ground floor scale is appropriate, maintaining good urban permeability."}

    impact_ratio = len(impacted_neighbors) / len(context_buildings) if context_buildings else 0
    if impact_ratio > 0.3:
        report['impact'] = {"score": "[-] SEVERE IMPACT", "desc": f"Significant obstruction of views and light for {len(impacted_neighbors)} neighboring buildings. Setbacks are strongly recommended."}
    elif len(impacted_neighbors) > 0:
        report['impact'] = {"score": "[/] MODERATE", "desc": f"Minor shadowing impact on {len(impacted_neighbors)} adjacent buildings, likely within acceptable zoning limits."}
    else:
        report['impact'] = {"score": "[+] OPTIMAL", "desc": "Massing integrates harmoniously with the context. No significant obstruction observed."}

    return total_volume, max_h, len(impacted_neighbors), report

# ==========================================
# 4. Visualization
# ==========================================
def plot_urban_scene(context_buildings, blocks=None):
    fig = go.Figure()
    
    if context_buildings:
        for b in context_buildings:
            pts = b['footprint']; h = b['height']
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            
            is_impacted = b.get('impacted', False)
            cg = 'rgba(255, 60, 60, 0.85)' if is_impacted else 'rgba(50, 50, 50, 0.5)'
            ce = '#FF4444' if is_impacted else '#555555'
            line_w = 2 if is_impacted else 1
            
            fig.add_trace(go.Scatter3d(x=x, y=y, z=[0]*len(pts), mode='lines', surfaceaxis=2, surfacecolor=cg, line=dict(color=ce, width=line_w), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=[h]*len(pts), mode='lines', surfaceaxis=2, surfacecolor=cg, line=dict(color=ce, width=line_w), showlegend=False, hoverinfo='skip'))
            for i in range(len(pts) - 1): fig.add_trace(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[0, h], mode='lines', showlegend=False, line=dict(color=ce, width=line_w), hoverinfo='skip'))
    
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers+text', marker=dict(color='#FFFFFF', size=3, symbol='cross'), text=["SITE"], textposition="top right", textfont=dict(color='#888888'), showlegend=False, hoverinfo='skip'))

    if blocks:
        pg = 'rgba(255, 255, 255, 0.95)'
        pe = '#FFFFFF' 
        for b in blocks:
            hw = b.width/2; hl = b.length/2; ox = b.offset_x; oy = b.offset_y; zmin = b.elevation; zmax = b.elevation + b.height
            x = [ox-hw, ox+hw, ox+hw, ox-hw, ox-hw]; y = [oy-hl, oy-hl, oy+hl, oy+hl, oy-hl]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=[zmin]*5, mode='lines', surfaceaxis=2, surfacecolor=pg, line=dict(color=pe, width=3), showlegend=False))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=[zmax]*5, mode='lines', surfaceaxis=2, surfacecolor=pg, line=dict(color=pe, width=3), showlegend=False))
            for i in range(4): fig.add_trace(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[zmin, zmax], mode='lines', showlegend=False, line=dict(color=pe, width=3)))
            
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ==========================================
# 5. UI & Business Logic Integration
# ==========================================
if 'location_name' not in st.session_state: st.session_state.location_name = "N/A"
if 'context_data' not in st.session_state: st.session_state.context_data = None
if 'proposal' not in st.session_state: st.session_state.proposal = None
if 'forma_connected' not in st.session_state: st.session_state.forma_connected = False

st.markdown("<h1>URBAN CONTEXTUAL GENERATIVE AGENT</h1>", unsafe_allow_html=True)

# --- Left Sidebar Controls ---
st.sidebar.markdown("### SYSTEM AUTH")
user_api_key = st.sidebar.text_input("OPENAI API KEY", type="password", placeholder="Enter key...")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("### 01. SITE CONTEXT")
address_input = st.sidebar.text_input("LOCATION:", value="Museum of Modern Art, NY")
r = st.sidebar.slider("SCAN RADIUS (m)", 50, 400, 200, 10)

if st.sidebar.button("EXTRACT URBAN FABRIC"):
    with st.spinner("Parsing geospatial data..."):
        lat, lon, name = geocode_address(address_input)
        if lat is not None:
            st.session_state.location_name = name
            st.session_state.context_data = fetch_urban_context(lat, lon, r)
            st.session_state.proposal = None
        else:
            st.sidebar.error("Geocoding failed.")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("### 02. ARCHITECTURAL INTENT")
intent = st.sidebar.text_area("PARAMETERS:", "A flagship art center. I want to break away from traditional closed blocks, creating cantilevered public spaces that engage with the street.")

if st.sidebar.button("EXECUTE GENERATION"):
    active_key = user_api_key if user_api_key else get_api_key("OPENAI_API_KEY")
    if not active_key:
        st.sidebar.error("MISSING API KEY.")
    elif not st.session_state.context_data or "error" in st.session_state.context_data:
        st.sidebar.error("PLEASE EXTRACT SITE FIRST.")
    else:
        with st.spinner("Synthesizing spatial topology..."):
            try:
                bldgs = st.session_state.context_data['buildings']
                context_str = f"Site surrounded by {len(bldgs)} buildings. Height range: {min([b['height'] for b in bldgs] + [0])}m to {max([b['height'] for b in bldgs] + [0])}m."
                st.session_state.proposal = generate_massing(active_key, context_str, intent)
                evaluate_urban_impact(st.session_state.proposal.blocks, st.session_state.context_data['buildings'])
            except Exception as e:
                st.sidebar.error(f"FAILED: {str(e)}")

# --- Main View & Post-Evaluation Panel ---
c1, c2 = st.columns([2.5, 1.2])

with c1:
    bldgs = st.session_state.context_data['buildings'] if st.session_state.context_data and "error" not in st.session_state.context_data else None
    blocks = st.session_state.proposal.blocks if st.session_state.proposal else None
    st.plotly_chart(plot_urban_scene(bldgs, blocks), use_container_width=True)

with c2:
    st.markdown("### DATA ANALYTICS")
    
    bldgs_count = len(st.session_state.context_data['buildings']) if st.session_state.context_data and "error" not in st.session_state.context_data else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">TARGET LOCATION</div>
        <div style="font-size: 0.9rem; color: #DDDDDD; margin-bottom: 10px;">{st.session_state.location_name}</div>
        <div class="metric-title">CONTEXT ENTITIES</div>
        <div class="metric-value">{bldgs_count}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.proposal:
        st.markdown("<div class='placeholder-box'>[ AWAITING TOPOLOGY GENERATION ]<br><br><span style='font-size:0.75rem; color:#888;'>Generate model to run environmental diagnostics.</span></div>", unsafe_allow_html=True)
    else:
        p = st.session_state.proposal
        vol, max_h, impacted_cnt, report = evaluate_urban_impact(p.blocks, st.session_state.context_data['buildings'])
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #FFFFFF;">
            <div class="metric-title" style="color:#FFFFFF;">ARCHITECT'S RATIONALE</div>
            <div class="rationale-text">{p.rationale}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="border-color: {'#FFFFFF' if 'OPTIMAL' in report['massing']['score'] or 'HUMAN' in report['massing']['score'] else '#FF5555'};">
            <div class="metric-title">MORPHOLOGY (EFFICIENCY)</div>
            <div class="eval-badge {'eval-good' if 'HUMAN' in report['massing']['score'] else 'eval-bad'}">{report['massing']['score']}</div>
            <div style="font-size: 0.85rem; color: #BBBBBB;">{report['massing']['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        impact_class = 'eval-bad' if 'SEVERE' in report['impact']['score'] else 'eval-warn' if 'MODERATE' in report['impact']['score'] else 'eval-good'
        border_color = '#FF5555' if 'SEVERE' in report['impact']['score'] else '#FFFFFF'
        
        st.markdown(f"""
        <div class="metric-card" style="border-color: {border_color};">
            <div class="metric-title">CONTEXTUAL IMPACT (RIGHT TO LIGHT)</div>
            <div class="eval-badge {impact_class}">{report['impact']['score']}</div>
            <div class="metric-value" style="font-size: 1.2rem; margin-bottom: 5px; color: {border_color};">{impacted_cnt} <span style="font-size:0.75rem; color:#BBBBBB;">NEIGHBORS IMPACTED</span></div>
            <div style="font-size: 0.85rem; color: #BBBBBB;">{report['impact']['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
