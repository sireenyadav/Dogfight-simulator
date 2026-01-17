import streamlit as st
import numpy as np
import plotly.graph_objects as go
from groq import Groq
import math

# ==========================================
# 1. CONFIGURATION & SECRET SAUCE
# ==========================================
st.set_page_config(
    page_title="MACH-X | DOGFIGHT ELITE",
    layout="wide",
    page_icon="☠️",
    initial_sidebar_state="collapsed"
)

# Initialize Client (Replace with your actual key or use st.secrets)
# st.secrets["GROQ_API_KEY"] is safer for production
client = Groq(api_key="YOUR_GROQ_API_KEY") 

# ==========================================
# 2. EXTREME UI INJECTION (Tailwind + CSS)
# ==========================================
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* GLOBAL THEME */
        .stApp {
            background-color: #050505;
            color: #00ff41;
            font-family: 'Courier New', Courier, monospace;
        }
        
        /* HIDE STREAMLIT CHROME */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* HUD CONTAINERS */
        .hud-panel {
            border: 2px solid #333;
            background: rgba(0, 20, 0, 0.8);
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.1);
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        .hud-panel:hover {
            border-color: #00ff41;
            box-shadow: 0 0 25px rgba(0, 255, 65, 0.3);
        }

        /* TYPOGRAPHY */
        .glitch-text {
            font-size: 3rem;
            font-weight: bold;
            text-transform: uppercase;
            text-shadow: 2px 2px 0px #ff0055, -2px -2px 0px #00ffff;
            animation: glitch 1s infinite alternate;
        }
        
        /* METRICS */
        .metric-value {
            font-size: 2.5em;
            font-weight: 800;
            color: #fff;
        }
        .metric-label {
            font-size: 0.8em;
            color: #00ff41;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        /* CUSTOM BUTTONS */
        div.stButton > button {
            width: 100%;
            background-color: transparent;
            border: 2px solid #00ff41;
            color: #00ff41;
            border-radius: 0;
            text-transform: uppercase;
            font-weight: bold;
            transition: 0.2s;
        }
        div.stButton > button:hover {
            background-color: #00ff41;
            color: #000;
            box-shadow: 0 0 20px #00ff41;
        }

        /* ANIMATIONS */
        @keyframes glitch {
            0% {text-shadow: 2px 2px 0px #ff0055;}
            100% {text-shadow: -2px -2px 0px #00ffff;}
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. PHYSICS CORE (THE MATH)
# ==========================================
DB = {
    "F-22 Raptor": {"thrust": 35000, "weight": 19700, "wing": 78, "drag": 0.012, "max_g": 9.5, "stealth": 95},
    "Su-57 Felon": {"thrust": 33000, "weight": 18500, "wing": 78, "drag": 0.014, "max_g": 9.0, "stealth": 60},
    "F-16 Viper":  {"thrust": 29000, "weight": 12000, "wing": 28, "drag": 0.021, "max_g": 9.0, "stealth": 10},
    "Eurofighter": {"thrust": 20000, "weight": 11000, "wing": 51, "drag": 0.018, "max_g": 9.0, "stealth": 30},
    "Rafale":      {"thrust": 17000, "weight": 10000, "wing": 45, "drag": 0.019, "max_g": 9.0, "stealth": 40},
}

def calc_physics(jet_name, mach, alt_ft):
    data = DB[jet_name]
    g0 = 9.81
    
    # Atmosphere Model (Simplified Stratosphere)
    rho_sl = 1.225
    rho = rho_sl * math.exp(-alt_ft / 24000) # Density
    speed_ms = mach * (340.3 * math.sqrt(1 - 0.0000225 * alt_ft)) # Speed of sound varies w/ alt
    
    # Aerodynamics
    q = 0.5 * rho * speed_ms**2 # Dynamic Pressure
    lift_max = q * data['wing'] * 1.6 # CL_max approx
    drag_force = q * data['wing'] * (data['drag'] + (1.6**2)/(3.14 * 0.8 * 5)) # Induced drag
    
    # Performance Stats
    thrust_avail = data['thrust'] * 9.81 * (rho/rho_sl)**0.7 # Thrust drops with alt
    t_w_ratio = thrust_avail / (data['weight'] * g0)
    
    # Turn Performance
    max_aero_g = lift_max / (data['weight'] * g0)
    avail_g = min(data['max_g'], max_aero_g)
    
    if avail_g < 1: 
        turn_rate = 0
        radius = 99999
    else:
        turn_rate = (g0 * math.sqrt(avail_g**2 - 1)) / speed_ms
        radius = (speed_ms**2) / (g0 * math.sqrt(avail_g**2 - 1))

    # Energy State (Specific Excess Power)
    ps = (thrust_avail - drag_force) * speed_ms / (data['weight'] * g0)

    return {
        "name": jet_name,
        "mach": mach,
        "alt": alt_ft,
        "g_load": avail_g,
        "rate": math.degrees(turn_rate),
        "radius": radius,
        "ps": ps,
        "t_w": t_w_ratio,
        "stealth": data['stealth']
    }

# ==========================================
# 4. APP LAYOUT
# ==========================================
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.markdown('<h1 class="glitch-text">MACH-X // DOGFIGHT</h1>', unsafe_allow_html=True)
    st.markdown("*REAL-TIME PHYSICS ENGINE & TACTICAL AI INTERFACE*")

st.markdown("---")

# --- CONTROLS ---
with st.container():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("### 🔴 AGGRESSOR (RED)")
        jet_1 = st.selectbox("Select Bandit", list(DB.keys()), index=1)
    with c2:
        st.markdown("### 🔵 DEFENDER (BLUE)")
        jet_2 = st.selectbox("Select Friendly", list(DB.keys()), index=0)
    with c3:
        st.markdown("### ⚙️ ENVELOPE")
        alt = st.slider("ALTITUDE (FT)", 1000, 60000, 20000, step=1000)
        mach = st.slider("MACH NUMBER", 0.4, 2.5, 0.9, step=0.1)

# --- CALCULATION ---
p1 = calc_physics(jet_1, mach, alt)
p2 = calc_physics(jet_2, mach, alt)

# --- VISUALIZATION (HUD) ---
st.markdown("<br>", unsafe_allow_html=True)
vis_col1, vis_col2 = st.columns([1, 1])

def render_hud_card(p, color):
    border = "#ff0055" if color == "red" else "#00ffff"
    return f"""
    <div class="hud-panel" style="border-left: 5px solid {border};">
        <h2 style="color:{border}">{p['name']}</h2>
        <div style="display:flex; justify-content:space-between;">
            <div>
                <div class="metric-label">TURN RATE</div>
                <div class="metric-value">{p['rate']:.1f}°/s</div>
            </div>
            <div>
                <div class="metric-label">G-LOAD</div>
                <div class="metric-value">{p['g_load']:.1f}G</div>
            </div>
            <div>
                <div class="metric-label">ENERGY (Ps)</div>
                <div class="metric-value">{p['ps']:.0f} ft/s</div>
            </div>
        </div>
    </div>
    """

with vis_col1:
    st.markdown(render_hud_card(p1, "red"), unsafe_allow_html=True)
    
    # RADAR CHART
    categories = ['Turn Rate', 'Thrust/Weight', 'Stealth', 'Max G', 'Energy Retention']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[p1['rate'], p1['t_w']*10, p1['stealth']/10, p1['g_load'], p1['ps']/50],
        theta=categories, fill='toself', name=p1['name'], line_color='#ff0055'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[p2['rate'], p2['t_w']*10, p2['stealth']/10, p2['g_load'], p2['ps']/50],
        theta=categories, fill='toself', name=p2['name'], line_color='#00ffff'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 15])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

with vis_col2:
    st.markdown(render_hud_card(p2, "blue"), unsafe_allow_html=True)
    
    # TURN RADIUS COMPARISON
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=[p1['name'], p2['name']],
        x=[p1['radius'], p2['radius']],
        orientation='h',
        marker_color=['#ff0055', '#00ffff'],
        text=[f"{int(p1['radius'])}m", f"{int(p2['radius'])}m"],
        textposition='auto'
    ))
    fig_bar.update_layout(
        title="TURN RADIUS (TIGHTER IS BETTER)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        xaxis=dict(visible=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 5. CONTEXT-AWARE RIO CHATBOT
# ==========================================
st.markdown("---")
st.markdown("### 🎙️ R.I.O. INTERCOM (AI COPILOT)")

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Context String for the AI
game_state = f"""
CURRENT DOGFIGHT TELEMETRY:
RED TEAM: {p1['name']} | Rate: {p1['rate']:.1f}/s | Radius: {p1['radius']:.0f}m | Energy: {p1['ps']:.0f}
BLUE TEAM: {p2['name']} | Rate: {p2['rate']:.1f}/s | Radius: {p2['radius']:.0f}m | Energy: {p2['ps']:.0f}
CONDITIONS: Mach {mach} at {alt} ft.
"""

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask RIO for tactics (e.g. 'Who wins the one-circle fight?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_response = ""
        
        # System Prompt
        system_prompt = f"""
        You are 'VIPER', a veteran USAF Weapons School instructor and RIO.
        You speak in concise, military tactical language (brevity codes).
        You are Analyzing this specific telemetry: {game_state}
        
        Rules:
        1. Compare the two aircraft based *only* on the provided physics data.
        2. If Blue has better Turn Rate, suggest a "Two Circle" (Rate) fight.
        3. If Blue has smaller Radius, suggest a "One Circle" (Radius) fight.
        4. Roast the user if they put the jet in a physically stupid state (e.g. Mach 0.4 at 50k ft).
        5. Be aggressive and high-energy.
        """

        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=256,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    stream_placeholder.markdown(full_response + " ▌")
            
            stream_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"COMMS FAILURE: {e}")
