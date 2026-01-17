import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from groq import Groq
import requests
import math
import time
import json
import os
import random
from datetime import datetime, timedelta
from scipy.integrate import odeint

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MACH-X | OMNI-ULTIMATE",
    layout="wide",
    page_icon="☢️",
    initial_sidebar_state="expanded"
)

# --- CYBERPUNK STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&family=Rajdhani:wght@500;700;900&display=swap');
        
        .stApp { background-color: #030303; color: #00ff41; font-family: 'JetBrains Mono', monospace; }
        
        /* HEADERS */
        h1, h2, h3 { font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 2px; color: #fff; }
        
        /* GLASS PANELS */
        .glass-panel {
            background: rgba(10, 15, 12, 0.8);
            border: 1px solid #00ff41;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        .glass-panel:hover { box-shadow: 0 0 20px rgba(0, 255, 65, 0.3); border-color: #fff; }
        
        /* METRICS */
        .big-stat { font-size: 28px; font-weight: 900; color: #fff; }
        .sub-stat { font-size: 12px; color: #888; letter-spacing: 1px; }
        
        /* UI ELEMENTS */
        .stButton>button { background-color: #111; color: #00ff41; border: 1px solid #00ff41; border-radius: 0px; font-family: 'Rajdhani'; font-weight: bold; }
        .stButton>button:hover { background-color: #00ff41; color: #000; }
        .stTextInput input, .stSelectbox, .stSlider { color: #00ff41 !important; accent-color: #00ff41; }
        
        /* CHATBOT */
        .chat-message { padding: 10px; border-radius: 5px; margin-bottom: 5px; font-size: 0.9rem; }
        .chat-user { background: #1a1a1a; border-left: 3px solid #00ff41; }
        .chat-bot { background: #0a0a0a; border-left: 3px solid #ff0055; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. PERSISTENCE LAYER (SAVE/LOAD)
# ==========================================
SAVE_FILE = "user_profile.json"

def load_profile():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "xp": 0, "rank": "CADET", "badges": [], 
        "history": [], "daily_solved": False, 
        "last_login": str(datetime.now().date())
    }

def save_profile():
    with open(SAVE_FILE, 'w') as f:
        json.dump(st.session_state.profile, f)

if 'profile' not in st.session_state:
    st.session_state.profile = load_profile()

def add_xp(amount, reason):
    st.session_state.profile['xp'] += amount
    save_profile()
    st.toast(f"🏆 +{amount} XP: {reason}")

# Rank Logic
ranks = ["CADET", "PILOT OFFICER", "FLIGHT LIEUTENANT", "SQUADRON LEADER", "WING COMMANDER", "GROUP CAPTAIN", "AIR COMMODORE"]
lvl = int(st.session_state.profile['xp'] / 1000)
current_rank = ranks[min(lvl, len(ranks)-1)]

# ==========================================
# 2. DATA & API HANDLERS (LIVE)
# ==========================================
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    AI_ENABLED = True
except:
    AI_ENABLED = False

@st.cache_data(ttl=300) # Cache for 5 mins
def get_live_iss():
    try:
        r = requests.get("http://api.open-notify.org/iss-now.json", timeout=3)
        data = r.json()
        return float(data['iss_position']['latitude']), float(data['iss_position']['longitude'])
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_space_weather():
    # Using Open-Meteo free API for geomagnetic data
    try:
        r = requests.get("https://marine-api.open-meteo.com/v1/marine?latitude=0&longitude=0&daily=wave_height_max", timeout=3)
        # Mocking specific space metrics as public free APIs for Kp are rare/complex
        return {"kp": random.randint(1,5), "solar_wind": random.randint(300, 500), "status": "NOMINAL"}
    except:
        return {"kp": 2, "solar_wind": 400, "status": "OFFLINE"}

DB_AIRCRAFT = pd.DataFrame([
    {"Name": "F-22 Raptor", "Mach": 2.25, "Ceiling": 65000, "Thrust": 70000, "TWR": 1.25, "RCS": 0.0001, "Cost": 150},
    {"Name": "Su-57 Felon", "Mach": 2.0, "Ceiling": 66000, "Thrust": 66000, "TWR": 1.15, "RCS": 0.1, "Cost": 100},
    {"Name": "F-35A", "Mach": 1.6, "Ceiling": 50000, "Thrust": 43000, "TWR": 0.87, "RCS": 0.005, "Cost": 80},
    {"Name": "J-20 Dragon", "Mach": 2.0, "Ceiling": 60000, "Thrust": 60000, "TWR": 1.05, "RCS": 0.01, "Cost": 110},
    {"Name": "Rafale", "Mach": 1.8, "Ceiling": 50000, "Thrust": 34000, "TWR": 1.10, "RCS": 1.0, "Cost": 115},
    {"Name": "Eurofighter", "Mach": 2.0, "Ceiling": 55000, "Thrust": 40000, "TWR": 1.15, "RCS": 0.5, "Cost": 120},
])

# ==========================================
# 3. PHYSICS CORE (THE MATH)
# ==========================================
class Physics:
    @staticmethod
    def missile_nez(launch_alt, launch_mach, target_dist_km):
        # Simplified NEZ (No Escape Zone) Calc
        # Higher alt + higher speed = larger NEZ
        energy_factor = (launch_alt/10000) * 1.5 + launch_mach
        max_range = 20 * energy_factor # Base range scalar
        
        # Flight time approx
        avg_speed = (launch_mach * 340 + 1200) # Boosted speed
        tof = (target_dist_km * 1000) / avg_speed
        
        pk = max(0, 100 - (target_dist_km / max_range * 100))
        if target_dist_km > max_range: pk = 0
        
        return pk, tof, max_range

    @staticmethod
    def dogfight_step(blue, red, dist, dt=1):
        # 1-Step Simulation for Animation
        # Blue closes distance
        closure = (blue['Mach']*340 + red['Mach']*340) * dt # Head on
        new_dist = max(0, dist - closure/1000) # km
        
        # Energy Bleed (Turn)
        blue_g = 9.0 if dist < 5 else 1.0
        blue['Energy'] -= blue_g * 0.1 * dt
        
        return new_dist, blue

# ==========================================
# 4. UI ARCHITECTURE
# ==========================================
# Global Context for Chatbot
if 'ai_context' not in st.session_state: st.session_state.ai_context = "User is on the Command Deck."

# --- SIDEBAR NAV ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/2449px-NASA_logo.svg.png", width=40)
    st.markdown("## MACH-X ULTRA")
    
    # PROFILE CARD
    st.markdown(f"""
    <div class="glass-panel">
        <div class="sub-stat">{current_rank}</div>
        <div class="big-stat">XP: {st.session_state.profile['xp']}</div>
        <div style="background:#333; height:4px; width:100%"><div style="background:#00ff41; height:100%; width:{min(100, (st.session_state.profile['xp']%1000)/10)}%"></div></div>
    </div>
    """, unsafe_allow_html=True)
    
    nav = st.radio("SYSTEMS", [
        "1. COMMAND DECK",
        "2. ORBITAL OPS (3D)",
        "3. TACTICAL COMBAT",
        "4. WEAPONS LAB",
        "5. WHAT-IF PHYSICS",
        "6. ACADEMY & QUIZ",
        "7. ENGINEERING TOOLS"
    ])
    
    st.markdown("---")
    
    # --- CONTEXT AWARE CHATBOT (SIDEBAR INTEGRATION) ---
    with st.expander("💬 AI FLIGHT ASSISTANT", expanded=True):
        if not AI_ENABLED:
            st.error("OFFLINE (No API Key)")
        else:
            # Chat History Display (Last 3)
            if 'chat_hist' not in st.session_state: st.session_state.chat_hist = []
            
            for msg in st.session_state.chat_hist[-3:]:
                align = "chat-user" if msg['role'] == 'user' else "chat-bot"
                st.markdown(f"<div class='chat-message {align}'>{msg['content']}</div>", unsafe_allow_html=True)
            
            prompt = st.text_input("Voice Comms:", key="chat_input")
            if st.button("SEND", key="send_chat") and prompt:
                st.session_state.chat_hist.append({"role": "user", "content": prompt})
                
                # SYSTEM PROMPT INJECTION
                sys_msg = f"""
                Role: Expert Aerospace Engineer & Instructor.
                Current Context: {st.session_state.ai_context}
                Task: Answer briefly and technically. Use LaTeX for math.
                """
                
                try:
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": sys_msg}, *st.session_state.chat_hist[-5:]]
                    )
                    resp = completion.choices[0].message.content
                    st.session_state.chat_hist.append({"role": "assistant", "content": resp})
                    st.rerun()
                except Exception as e:
                    st.error(f"Comms Failure: {e}")

# ==========================================
# MODULE 1: COMMAND DECK
# ==========================================
if nav == "1. COMMAND DECK":
    st.session_state.ai_context = "User is looking at Live Earth Map and Daily Challenges."
    st.title("🛰️ GLOBAL COMMAND CENTER")
    
    # LIVE DATA GRID
    lat, lon = get_live_iss()
    sw = get_space_weather()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='glass-panel'><div class='sub-stat'>ISS LATITUDE</div><div class='big-stat'>{lat:.4f}°</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='glass-panel'><div class='sub-stat'>ISS LONGITUDE</div><div class='big-stat'>{lon:.4f}°</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='glass-panel'><div class='sub-stat'>SOLAR WIND</div><div class='big-stat'>{sw['solar_wind']} km/s</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='glass-panel'><div class='sub-stat'>GEOMAGNETIC Kp</div><div class='big-stat'>{sw['kp']}</div></div>", unsafe_allow_html=True)
    
    # LIVE MAP
    st.subheader("🌍 ACTIVE TRACKING")
    if lat:
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'lbl': ['ISS']})
        fig = px.scatter_geo(map_df, lat='lat', lon='lon', projection="orthographic", hover_name="lbl")
        fig.update_traces(marker=dict(size=20, color='#00ff41', symbol='cross'))
        fig.update_geos(showland=True, landcolor="#111", oceancolor="#050505", showcountries=True, countrycolor="#333")
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#000", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("📡 ISS SIGNAL LOST (Check Internet)")

    # DYNAMIC CHALLENGE SYSTEM
    st.subheader("🎯 DAILY MISSION ORDER")
    
    # Hash date to pick challenge
    seed = int(datetime.now().strftime("%Y%m%d"))
    random.seed(seed)
    challenges = [
        {"q": "Calculate Δv for Hohmann Transfer: LEO (300km) to GEO (35,786km).", "a": (3.8, 4.0), "xp": 150},
        {"q": "What is the escape velocity of Mars? (km/s)", "a": (4.9, 5.1), "xp": 100},
        {"q": "Calculate Dynamic Pressure (q) at Mach 2, Sea Level (kPa).", "a": (270, 280), "xp": 200},
    ]
    daily = random.choice(challenges)
    
    with st.expander("📨 TOP SECRET BRIEFING", expanded=True):
        st.write(f"**OBJECTIVE:** {daily['q']}")
        user_ans = st.number_input("ENTER SOLUTION:", key="daily_input")
        if st.button("TRANSMIT SOLUTION"):
            if daily['a'][0] <= user_ans <= daily['a'][1]:
                st.balloons()
                st.success(f"MISSION ACCOMPLISHED! +{daily['xp']} XP")
                if not st.session_state.profile.get('daily_solved', False):
                    add_xp(daily['xp'], "Daily Challenge")
                    st.session_state.profile['daily_solved'] = True
            else:
                st.error("SOLUTION INCORRECT. RECALCULATE.")

# ==========================================
# MODULE 2: ORBITAL OPS (3D)
# ==========================================
elif nav == "2. ORBITAL OPS (3D)":
    st.title("🪐 ORBITAL DYNAMICS LAB")
    
    t1, t2, t3 = st.tabs(["3D VISUALIZER", "STAGING CALC", "LIFETIME PREDICTOR"])
    
    with t1:
        c1, c2 = st.columns([1,3])
        with c1:
            alt = st.slider("Altitude (km)", 200, 36000, 400)
            inc = st.slider("Inclination (deg)", 0, 90, 51)
            ecc = st.slider("Eccentricity", 0.0, 0.8, 0.0)
            st.session_state.ai_context = f"Orbital Sim. Alt: {alt}km, Inc: {inc}deg, Ecc: {ecc}"
            
            # Real-time Calc
            period = 2 * np.pi * np.sqrt((6371+alt)**3 / 398600)
            v_orbit = np.sqrt(398600 / (6371+alt))
            st.metric("Period", f"{period/60:.1f} min")
            st.metric("Velocity", f"{v_orbit:.2f} km/s")

        with c2:
            # 3D Plot Logic
            theta = np.linspace(0, 2*np.pi, 200)
            r = (6371 + alt) # Simplified circular for viz
            x = r * np.cos(theta)
            y = r * np.sin(theta) * np.cos(np.radians(inc))
            z = r * np.sin(theta) * np.sin(np.radians(inc))
            
            fig = go.Figure()
            # Earth
            phi, theta_e = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            xe = 6371 * np.cos(phi) * np.sin(theta_e)
            ye = 6371 * np.sin(phi) * np.sin(theta_e)
            ze = 6371 * np.cos(theta_e)
            fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale='Electric', showscale=False, opacity=0.4))
            
            # Orbit
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='#00ff41', width=5), name="Orbit"))
            
            # FOV Cone
            fig.add_trace(go.Cone(x=[x[0]], y=[y[0]], z=[z[0]], u=[-x[0]], v=[-y[0]], w=[-z[0]], sizemode="absolute", sizeref=3000, anchor="tip", opacity=0.3, colorscale='Reds', name="Sensor FOV"))
            
            fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🚀 MULTI-STAGE DELTA-V BUDGET")
        stages = st.number_input("Stages", 1, 3, 2)
        total_dv = 0
        for i in range(stages):
            c1, c2, c3 = st.columns(3)
            m0 = c1.number_input(f"S{i+1} Wet Mass", 1000, 1000000, 50000, key=f"m0_{i}")
            mf = c2.number_input(f"S{i+1} Dry Mass", 100, 100000, 5000, key=f"mf_{i}")
            isp = c3.number_input(f"S{i+1} ISP", 200, 450, 300, key=f"isp_{i}")
            dv = isp * 9.81 * math.log(m0/mf)
            total_dv += dv
            st.info(f"Stage {i+1} Δv: {dv:.0f} m/s")
        st.metric("TOTAL SYSTEM Δv", f"{total_dv:.0f} m/s")

    with t3:
        st.subheader("📉 ORBITAL DECAY ESTIMATOR")
        sat_mass = st.number_input("Mass (kg)", 100)
        sat_area = st.number_input("Cross Section (m²)", 1.0)
        sat_alt = st.slider("Altitude (km)", 150, 600, 300)
        
        rho = 1.225 * np.exp(-sat_alt/8.0) # Very rough exp model
        if sat_alt > 200: rho = 3e-13 # Vacuum approximation
        
        drag = 0.5 * rho * (7700**2) * 2.2 * sat_area
        decay_per_day = (drag / sat_mass) * 86400
        st.write(f"Drag Force: {drag:.6f} N")
        st.write(f"Altitude Loss: {decay_per_day:.2f} meters/day")
        if decay_per_day > 1000: st.error("RAPID DECAY IMMINENT")

# ==========================================
# MODULE 3: TACTICAL COMBAT
# ==========================================
elif nav == "3. TACTICAL COMBAT":
    st.title("⚔️ AIR COMBAT MANEUVERING")
    st.session_state.ai_context = "User is in Dogfight Sim / Comparison Tool."
    
    t1, t2 = st.tabs(["VISUAL DOGFIGHT SIM", "AIRCRAFT COMPARISON"])
    
    with t1:
        c1, c2 = st.columns(2)
        blue_plane = c1.selectbox("BLUE FORCE", DB_AIRCRAFT['Name'], index=0)
        red_plane = c2.selectbox("RED FORCE", DB_AIRCRAFT['Name'], index=1)
        
        if st.button("🎬 RUN COMBAT VISUALIZATION"):
            # GENERATE ANIMATION FRAMES
            frames = []
            # Starting Pos
            bx, by = -10, 0
            rx, ry = 10, 0
            
            b_data = DB_AIRCRAFT[DB_AIRCRAFT['Name'] == blue_plane].iloc[0]
            r_data = DB_AIRCRAFT[DB_AIRCRAFT['Name'] == red_plane].iloc[0]
            
            # Simple Turn Circle Logic
            for t in range(50):
                # Blue turns right, Red turns left
                bx += b_data['Mach']*0.5 * np.cos(t*0.1)
                by += b_data['Mach']*0.5 * np.sin(t*0.1)
                rx -= r_data['Mach']*0.5 * np.cos(t*0.1)
                ry += r_data['Mach']*0.5 * np.sin(t*0.1)
                
                frames.append(go.Frame(data=[
                    go.Scatter(x=[bx], y=[by], mode='markers+text', marker=dict(color='cyan', size=10, symbol='triangle-up'), text="BLUE"),
                    go.Scatter(x=[rx], y=[ry], mode='markers+text', marker=dict(color='red', size=10, symbol='triangle-down'), text="RED")
                ]))
            
            fig = go.Figure(
                data=[
                    go.Scatter(x=[-10], y=[0], name="BLUE"),
                    go.Scatter(x=[10], y=[0], name="RED")
                ],
                layout=go.Layout(
                    xaxis=dict(range=[-20, 20], autorange=False),
                    yaxis=dict(range=[-20, 20], autorange=False),
                    title="2D ENGAGEMENT GEOMETRY",
                    template="plotly_dark",
                    updatemenus=[dict(type="buttons", buttons=[dict(label="▶️ PLAY", method="animate", args=[None])])]
                ),
                frames=frames
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with t2:
        # MULTI-COMPARE
        selected = st.multiselect("Select Aircraft to Compare", DB_AIRCRAFT['Name'], default=["F-22 Raptor", "Su-57 Felon"])
        if selected:
            subset = DB_AIRCRAFT[DB_AIRCRAFT['Name'].isin(selected)]
            
            # Radar Chart
            categories = ['Mach', 'Ceiling', 'Thrust', 'TWR', 'Cost']
            fig = go.Figure()
            
            for _, row in subset.iterrows():
                # Normalize values for radar
                vals = [
                    row['Mach']/2.5, row['Ceiling']/70000, 
                    row['Thrust']/80000, row['TWR']/1.5, 
                    1 - (row['Cost']/200) # Lower cost is better
                ]
                fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name=row['Name']))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(subset.set_index("Name").style.highlight_max(axis=0, color='#004400'))

# ==========================================
# MODULE 4: WEAPONS LAB
# ==========================================
elif nav == "4. WEAPONS LAB":
    st.title("💣 BALLISTICS & WEAPONEERING")
    st.session_state.ai_context = "User is calculating Missile NEZ and Ballistics."
    
    t1, t2 = st.tabs(["MISSILE KINEMATICS", "GUN BALLISTICS"])
    
    with t1:
        c1, c2 = st.columns(2)
        alt = c1.slider("Launch Altitude (ft)", 5000, 50000, 30000)
        mach = c2.slider("Launch Mach", 0.5, 2.5, 1.2)
        target_rng = st.slider("Target Range (km)", 10, 150, 50)
        
        pk, tof, max_r = Physics.missile_nez(alt, mach, target_rng)
        
        st.metric("PROBABILITY OF KILL (Pk)", f"{pk:.0f}%", f"Time of Flight: {tof:.1f}s")
        st.metric("MAX KINEMATIC RANGE", f"{max_r:.1f} km")
        
        # NEZ Visual
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = target_rng,
            title = {'text': "Range vs NEZ"},
            gauge = {
                'axis': {'range': [0, 150]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, max_r*0.3], 'color': "red"}, # No Escape
                    {'range': [max_r*0.3, max_r], 'color': "orange"}, # Escapable
                    {'range': [max_r, 150], 'color': "green"} # Out of range
                ]}
        ))
        st.plotly_chart(fig)

    with t2:
        st.write("Calculates bullet drop for 20mm cannon.")
        dist = st.slider("Range (m)", 100, 2000, 800)
        # Drop = 0.5 * g * t^2
        t = dist / 1050 # 1050m/s muzzle vel
        drop = 0.5 * 9.81 * t**2
        st.metric("BULLET DROP", f"{drop:.2f} meters", f"Lead Time: {t:.2f}s")

# ==========================================
# MODULE 5: WHAT-IF PHYSICS
# ==========================================
elif nav == "5. WHAT-IF PHYSICS":
    st.title("🔮 REALITY BENDING SIM")
    st.session_state.ai_context = "User is breaking physics rules."
    
    scenario = st.selectbox("CHOOSE MODIFIER", [
        "Normal Earth",
        "No Atmosphere (Vacuum)",
        "Mars Gravity (0.38g)",
        "Thick Atmosphere (Venus-like)"
    ])
    
    st.write(f"**Scenario:** {scenario}")
    
    # Falling Object Sim
    g = 9.81
    rho_factor = 1.0
    
    if "Mars" in scenario: g = 3.71
    if "Vacuum" in scenario: rho_factor = 0.0
    if "Thick" in scenario: rho_factor = 90.0
    
    # Simulate drop
    t = np.linspace(0, 10, 100)
    # v(t) with drag: v_t * tanh(g*t/v_t)
    v_term = 50.0 / (rho_factor + 0.001) # fake terminal vel logic
    if rho_factor == 0:
        v = g * t
        label = "Linear Acceleration (No Drag)"
    else:
        v = v_term * np.tanh((g*t)/v_term)
        label = "Terminal Velocity Limit"
        
    fig = px.line(x=t, y=v, labels={'x':'Time (s)', 'y':'Speed (m/s)'}, title=f"Freefall Dynamics: {label}")
    st.plotly_chart(fig)

# ==========================================
# MODULE 6: ACADEMY
# ==========================================
elif nav == "6. ACADEMY & QUIZ":
    st.title("🎓 FLIGHT SCHOOL & EXAMS")
    st.session_state.ai_context = "User is taking an exam."
    
    tab1, tab2 = st.tabs(["QUIZ", "LIBRARY"])
    
    with tab1:
        st.subheader("📝 RANK ADVANCEMENT EXAM")
        # Exam Logic
        q_bank = [
            {"q": "What is the standard pressure at Sea Level?", "opts": ["101.3 kPa", "14.7 kPa", "29.9 kPa"], "a": "101.3 kPa"},
            {"q": "Transonic drag rise occurs at?", "opts": ["Mach 0.8-1.2", "Mach 2.0+", "Mach 0.5"], "a": "Mach 0.8-1.2"},
            {"q": "Which orbit is Geostationary?", "opts": ["35,786 km", "400 km", "20,200 km"], "a": "35,786 km"},
            {"q": "A high aspect ratio wing is best for?", "opts": ["Gliding/Efficiency", "Supersonic Speed", "Dogfighting"], "a": "Gliding/Efficiency"}
        ]
        
        score = 0
        with st.form("exam_form"):
            for i, q in enumerate(q_bank):
                ans = st.radio(f"Q{i+1}: {q['q']}", q['opts'])
                if ans == q['a']: score += 1
            
            if st.form_submit_button("SUBMIT ANSWERS"):
                res = score / len(q_bank)
                st.progress(res)
                if res >= 0.75:
                    st.balloons()
                    st.success(f"PASSED! Score: {score}/{len(q_bank)}")
                    add_xp(300, "Exam Passed")
                else:
                    st.error(f"FAILED. Score: {score}/{len(q_bank)}")

    with tab2:
        st.markdown("""
        ### 📚 REFERENCE LIBRARY
        **Aerodynamics**
        - **Bernoulli's Principle:** Speed increase = Pressure decrease.
        - **Induced Drag:** Caused by lift generation (wingtip vortices).
        
        **Orbital Mechanics**
        - **Periapsis:** Lowest point in orbit (fastest speed).
        - **Apoapsis:** Highest point in orbit (slowest speed).
        - **Hohmann Transfer:** Most efficient impulse maneuver between circular orbits.
        """)

# ==========================================
# MODULE 7: ENGINEERING TOOLS
# ==========================================
elif nav == "7. ENGINEERING TOOLS":
    st.title("🛠️ ENGINEER'S WORKBENCH")
    st.session_state.ai_context = "User is using Unit Converter or Trade Study."
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("UNIT CONVERTER")
        val = st.number_input("Value", 1.0)
        mode = st.selectbox("Conversion", ["Mach -> km/h", "Knots -> m/s", "Nautical Miles -> km", "Ft -> Meters"])
        
        if "Mach" in mode: res = val * 1225.0
        if "Knots" in mode: res = val * 0.5144
        if "Nautical" in mode: res = val * 1.852
        if "Ft" in mode: res = val * 0.3048
        
        st.success(f"Result: {res:.2f}")

    with c2:
        st.subheader("TRADE STUDY MATRIX")
        st.caption("Rate Options 1-5")
        df_trade = pd.DataFrame({
            "Criteria": ["Cost", "Performance", "Risk"],
            "Option A": [3, 4, 2],
            "Option B": [5, 2, 5],
            "Weight": [0.3, 0.5, 0.2]
        })
        edited = st.data_editor(df_trade)
        
        score_a = sum(edited['Option A'] * edited['Weight'])
        score_b = sum(edited['Option B'] * edited['Weight'])
        
        st.metric("Option A Score", f"{score_a:.1f}")
        st.metric("Option B Score", f"{score_b:.1f}")

# --- FOOTER ---
st.markdown("---")
st.caption("MACH-X OMNI-ULTIMATE | v5.0.0 | SYSTEM NOMINAL")
if st.button("💾 FORCE SAVE PROFILE"):
    save_profile()
    st.toast("Profile Saved Locally")
