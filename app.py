import streamlit as st
import numpy as np
import plotly.graph_objects as go
from groq import Groq
import math
import json

# ==========================================
# 1. CONFIGURATION & SECRET SAUCE
# ==========================================
st.set_page_config(
    page_title="MACH-X | OMNI-CRUNCH",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Initialize Client
# SAFETY CHECK: If no key is found, don't crash, just warn.
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("🚨 CRITICAL: GROQ_API_KEY not found in Secrets! Add it in Streamlit Dashboard.")
    st.stop()

# ==========================================
# 2. EXTREME UI (Tailwind + Glassmorphism)
# ==========================================
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .stApp { background-color: #000000; color: #00ff41; font-family: 'Courier New', monospace; }
        .block-container { padding-top: 1rem; }
        
        /* NEON GLASS CARDS */
        .glass-panel {
            background: rgba(10, 20, 10, 0.7);
            border: 1px solid #33ff33;
            box-shadow: 0 0 15px rgba(51, 255, 51, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* INPUT FIELDS */
        .stTextInput input {
            background-color: #111;
            color: #00ff41;
            border: 1px solid #333;
        }
        
        /* METRICS */
        .stat-value { font-size: 2.2rem; font-weight: 900; text-shadow: 0 0 10px rgba(0,255,65,0.5); }
        .stat-label { font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: #888; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #111;
            border: 1px solid #333;
            color: #fff;
            border-radius: 4px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00ff41 !important;
            color: #000 !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. GENERATIVE PHYSICS ENGINE (The Brains)
# ==========================================
@st.cache_data
def get_vehicle_stats(vehicle_name, mode):
    """
    Uses Groq to 'Hallucinate' accurate physics data for ANY vehicle.
    Returns JSON.
    """
    if mode == "AERO":
        prompt = f"""
        Return a valid JSON object (NO TEXT, JUST JSON) for the aircraft '{vehicle_name}'.
        Estimate these exact physics values based on real-world specs:
        {{
            "thrust": (Total thrust in Newtons, integer),
            "weight": (Combat weight in kg, integer),
            "wing_area": (Wing area in m^2, float),
            "drag_coeff": (Parasitic drag coefficient, approx 0.01-0.03, float),
            "max_g": (Max structural G-load, float),
            "stealth": (0-100 score, 100=invisible),
            "name": "{vehicle_name}"
        }}
        """
    else: # ROCKET MODE
        prompt = f"""
        Return a valid JSON object (NO TEXT, JUST JSON) for the rocket/spacecraft '{vehicle_name}'.
        Estimate these exact values:
        {{
            "thrust_sl": (Sea Level Thrust in Newtons, integer),
            "thrust_vac": (Vacuum Thrust in Newtons, integer),
            "mass_wet": (Launch mass in kg, integer),
            "mass_dry": (Empty mass in kg, integer),
            "isp_sl": (Specific Impulse Sea Level, seconds, integer),
            "isp_vac": (Specific Impulse Vacuum, seconds, integer),
            "stages": (Number of stages, integer),
            "name": "{vehicle_name}"
        }}
        """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an aerospace engineering database. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        # Clean the response to ensure it's pure JSON
        content = completion.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except:
        return None

# ==========================================
# 4. PHYSICS MATH (The Calculator)
# ==========================================
def calc_aero_physics(stats, mach, alt_ft):
    # Standard Atmosphere
    g0 = 9.81
    rho = 1.225 * math.exp(-alt_ft / 24000)
    speed_ms = mach * (340.3 * math.sqrt(1 - 0.0000225 * alt_ft))
    
    # Dynamics
    q = 0.5 * rho * speed_ms**2
    lift_max = q * stats['wing_area'] * 1.5
    drag = q * stats['wing_area'] * stats['drag_coeff']
    
    thrust = stats['thrust'] * ((rho/1.225)**0.6) # Thrust lapse
    
    # Performance
    turn_rate = 0
    radius = 0
    avail_g = lift_max / (stats['weight'] * g0)
    functional_g = min(stats['max_g'], avail_g)
    
    if functional_g > 1:
        turn_rate = math.degrees((g0 * math.sqrt(functional_g**2 - 1)) / speed_ms)
        radius = (speed_ms**2) / (g0 * math.sqrt(functional_g**2 - 1))
        
    ps = (thrust - drag) * speed_ms / (stats['weight'] * g0)
    
    return {
        "g": functional_g, 
        "rate": turn_rate, 
        "radius": radius, 
        "ps": ps, 
        "mach": mach
    }

def calc_rocket_physics(stats):
    # Tsiolkovsky Rocket Equation
    g0 = 9.81
    delta_v = stats['isp_vac'] * g0 * math.log(stats['mass_wet'] / stats['mass_dry'])
    twr = stats['thrust_sl'] / (stats['mass_wet'] * g0)
    burn_time = (stats['isp_vac'] * g0 * (stats['mass_wet'] - stats['mass_dry'])) / stats['thrust_vac']
    
    return {"dv": delta_v, "twr": twr, "burn": burn_time}

# ==========================================
# 5. THE APP INTERFACE
# ==========================================
st.markdown("## ⚡ MACH-X // OMNI-LAB")

# MODE SELECTION
mode = st.radio("SELECT SIMULATION MODE", ["✈️ ATMOSPHERIC COMBAT", "🚀 ORBITAL MECHANICS"], horizontal=True)

if mode == "✈️ ATMOSPHERIC COMBAT":
    # ---------------- COMBAT MODE ----------------
    c1, c2 = st.columns(2)
    with c1:
        v1_name = st.text_input("RED AIRCRAFT", "Su-57 Felon")
        if st.button("LOAD RED DATA"):
            st.session_state['v1_stats'] = get_vehicle_stats(v1_name, "AERO")
            
    with c2:
        v2_name = st.text_input("BLUE AIRCRAFT", "F-22 Raptor")
        if st.button("LOAD BLUE DATA"):
            st.session_state['v2_stats'] = get_vehicle_stats(v2_name, "AERO")
    
    # FLIGHT ENVELOPE CONTROLS
    st.markdown("### 🕹️ FLIGHT ENVELOPE")
    ec1, ec2 = st.columns(2)
    mach = ec1.slider("MACH", 0.5, 3.0, 0.9)
    alt = ec2.slider("ALTITUDE (FT)", 1000, 60000, 20000)

    # PROCESS & DISPLAY
    if 'v1_stats' in st.session_state and 'v2_stats' in st.session_state:
        p1 = calc_aero_physics(st.session_state['v1_stats'], mach, alt)
        p2 = calc_aero_physics(st.session_state['v2_stats'], mach, alt)
        
        # HUD 
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown(f"""
            <div class="glass-panel" style="border-left: 5px solid #ff0055;">
                <h3 style="color:#ff0055">{st.session_state['v1_stats']['name']}</h3>
                <div class="stat-value">{p1['rate']:.1f}°/s</div><div class="stat-label">TURN RATE</div>
                <div class="stat-value">{p1['g']:.1f}G</div><div class="stat-label">MAX LOAD</div>
                <div class="stat-value">{p1['ps']:.0f} ft/s</div><div class="stat-label">ENERGY (Ps)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with hc2:
            st.markdown(f"""
            <div class="glass-panel" style="border-left: 5px solid #00ffff;">
                <h3 style="color:#00ffff">{st.session_state['v2_stats']['name']}</h3>
                <div class="stat-value">{p2['rate']:.1f}°/s</div><div class="stat-label">TURN RATE</div>
                <div class="stat-value">{p2['g']:.1f}G</div><div class="stat-label">MAX LOAD</div>
                <div class="stat-value">{p2['ps']:.0f} ft/s</div><div class="stat-label">ENERGY (Ps)</div>
            </div>
            """, unsafe_allow_html=True)
            
        # CHART: TURN RATE VS MACH
        x_mach = np.linspace(0.5, 2.5, 20)
        y1_rate = [calc_aero_physics(st.session_state['v1_stats'], m, alt)['rate'] for m in x_mach]
        y2_rate = [calc_aero_physics(st.session_state['v2_stats'], m, alt)['rate'] for m in x_mach]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_mach, y=y1_rate, name="RED TEAM", line=dict(color="#ff0055", width=4)))
        fig.add_trace(go.Scatter(x=x_mach, y=y2_rate, name="BLUE TEAM", line=dict(color="#00ffff", width=4)))
        fig.update_layout(title="TURN RATE PERFORMANCE VS MACH", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

else:
    # ---------------- ROCKET MODE ----------------
    st.markdown("### 🪐 ORBITAL LAUNCH ANALYZER")
    r_name = st.text_input("ENTER ROCKET NAME (e.g., Saturn V, Falcon 9)", "Falcon 9")
    
    if st.button("RUN LAUNCH SIMULATION"):
        stats = get_vehicle_stats(r_name, "ROCKET")
        if stats:
            p = calc_rocket_physics(stats)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='glass-panel'><div class='stat-value'>{p['dv']:.0f} m/s</div><div class='stat-label'>TOTAL DELTA-V</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='glass-panel'><div class='stat-value'>{p['twr']:.2f}</div><div class='stat-label'>TWR (LIFTOFF)</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='glass-panel'><div class='stat-value'>{p['burn']:.0f} s</div><div class='stat-label'>BURN TIME</div></div>", unsafe_allow_html=True)
            
            # VISUALIZATION
            fig_r = go.Figure(go.Bar(
                x=[p['dv'], 9400],
                y=[stats['name'], "Orbit Req."],
                orientation='h',
                marker_color=['#00ff41', '#555']
            ))
            fig_r.update_layout(title="CAPABILITY VS ORBIT REQUIREMENT (LEO)", template="plotly_dark")
            st.plotly_chart(fig_r)

# ==========================================
# 6. UNIVERSAL AI CHATBOT
# ==========================================
st.markdown("---")
st.markdown("### 📡 ENGINEERING ASSISTANT")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the physics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    context = "USER IS IN ROCKET MODE" if mode == "🚀 ORBITAL MECHANICS" else "USER IS IN COMBAT MODE"
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are a Physics Expert. Context: {context}. Keep it brief and technical."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
        
