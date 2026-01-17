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
import random
from datetime import datetime, timedelta
from scipy.integrate import odeint

# ==========================================
# 0. CORE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="MACH-X | OMNI-CRUNCH ULTIMATE",
    layout="wide",
    page_icon="☢️",
    initial_sidebar_state="collapsed"
)

# --- CYBERPUNK CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&family=Rajdhani:wght@500;700&display=swap');
        
        .stApp { background-color: #050505; color: #00ff41; font-family: 'JetBrains Mono', monospace; }
        h1, h2, h3, h4 { font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 2px; color: #fff; }
        
        .glass-panel {
            background: rgba(15, 20, 15, 0.85);
            border: 1px solid #00ff41;
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.1);
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .stat-val { font-size: 24px; font-weight: 800; color: #00ff41; }
        .stat-lbl { font-size: 10px; color: #888; letter-spacing: 1px; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { background-color: #111; border-radius: 0px; border-bottom: 1px solid #333; }
        .stTabs [aria-selected="true"] { border-bottom: 2px solid #00ff41; color: #00ff41 !important; }
        
        /* PROGRESS BAR */
        .stProgress > div > div > div > div { background-color: #00ff41; }
    </style>
""", unsafe_allow_html=True)

# --- API INIT ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    AI_ENABLED = True
except:
    AI_ENABLED = False

# ==========================================
# 1. STATE MANAGEMENT & GAMIFICATION
# ==========================================
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        "xp": 1250,
        "rank": "FLIGHT LIEUTENANT",
        "badges": ["First Flight"],
        "missions_completed": 4,
        "streak": 3
    }
if 'history' not in st.session_state: st.session_state.history = []
if 'messages' not in st.session_state: st.session_state.messages = []

def add_xp(amount, reason):
    st.session_state.user_profile['xp'] += amount
    st.toast(f"🏆 +{amount} XP: {reason}")

# ==========================================
# 2. PHYSICS KERNEL (THE BRAIN)
# ==========================================
class PhysicsCore:
    @staticmethod
    def get_atmosphere(alt_m):
        """Standard Atmosphere 1976"""
        if alt_m > 80000: return 0.0, 0.0
        T = 288.15 - 0.0065 * alt_m if alt_m < 11000 else 216.65
        P = 101325 * (1 - 0.0065 * alt_m / 288.15)**5.25 if alt_m < 11000 else 22632 * math.exp(-9.81 * (alt_m - 11000) / (287 * 216.65))
        rho = P / (287 * T)
        return rho

    @staticmethod
    def ballistics_trajectory(v0, angle_deg, drag_coeff, area, mass):
        """Projectile Motion with Drag (Euler Method)"""
        dt = 0.1
        g = 9.81
        angle_rad = math.radians(angle_deg)
        vx = v0 * math.cos(angle_rad)
        vy = v0 * math.sin(angle_rad)
        x, y = 0, 0
        traj_x, traj_y = [0], [0]
        
        while y >= 0:
            v = math.sqrt(vx**2 + vy**2)
            rho = PhysicsCore.get_atmosphere(y)
            drag = 0.5 * rho * v**2 * drag_coeff * area
            ax = -(drag * (vx/v)) / mass
            ay = -g - (drag * (vy/v)) / mass
            
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            
            if y < 0: break
            traj_x.append(x)
            traj_y.append(y)
            
        return traj_x, traj_y

    @staticmethod
    def propagate_orbit(altitude_km, inclination_deg, eccentricity=0):
        """Keplerian Propagator"""
        mu = 3.986e5 # Earth GM
        R_e = 6371
        a = R_e + altitude_km
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        
        # Generate Ground Track
        time_steps = np.linspace(0, period, 100)
        mean_motion = np.sqrt(mu / a**3)
        
        lats, lons, x, y, z = [], [], [], [], []
        
        for t in time_steps:
            # Simplified circular orbit logic for ground track
            theta = mean_motion * t
            
            # 3D Coordinates (Inertial)
            x_i = a * np.cos(theta)
            y_i = a * np.sin(theta) * np.cos(np.radians(inclination_deg))
            z_i = a * np.sin(theta) * np.sin(np.radians(inclination_deg))
            x.append(x_i); y.append(y_i); z.append(z_i)
            
            # Lat/Lon (Rotating Earth)
            lat = np.degrees(np.arcsin(z_i / a))
            # Earth rotation correction (15 deg/hour)
            earth_rot = (t / 3600) * 15
            lon = np.degrees(np.arctan2(y_i, x_i)) - earth_rot
            
            # Normalize Lon -180 to 180
            lon = (lon + 180) % 360 - 180
            
            lats.append(lat)
            lons.append(lon)
            
        return lats, lons, x, y, z, period

# ==========================================
# 3. LIVE DATA MOCKER (API HANDLERS)
# ==========================================
def get_iss_location():
    try:
        r = requests.get("http://api.open-notify.org/iss-now.json", timeout=2)
        data = r.json()
        return float(data['iss_position']['latitude']), float(data['iss_position']['longitude'])
    except:
        # Fallback simulation if API fails
        t = time.time()
        return 51.6 * math.sin(t/1000), (t/100) % 360 - 180

def get_space_weather():
    # Mocked because NOAA SWPC requires API key complexity
    return {
        "solar_wind": random.randint(300, 600), # km/s
        "kp_index": random.randint(0, 9),
        "sunspots": random.randint(10, 150)
    }

# ==========================================
# 4. DASHBOARD UI
# ==========================================

# --- SIDEBAR: PROFILE & NAV ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/2449px-NASA_logo.svg.png", width=50)
    st.markdown("## MACH-X OMNI")
    
    # User Card
    st.markdown(f"""
    <div class="glass-panel">
        <h4 style='margin:0'>{st.session_state.user_profile['rank']}</h4>
        <div style='display:flex; justify-content:space-between'>
            <span>XP: {st.session_state.user_profile['xp']}</span>
            <span>LVL: {int(st.session_state.user_profile['xp']/500)}</span>
        </div>
        <div style='height:4px; width:100%; background:#333; margin-top:5px'>
            <div style='height:100%; width:{st.session_state.user_profile['xp']%500/5}%; background:#00ff41'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    nav = st.radio("SYSTEMS", [
        "COMMAND DECK",
        "SIM: ORBITAL",
        "SIM: ATMOSPHERIC",
        "WEAPONS & BALLISTICS",
        "ENGINEERING TOOLS",
        "ACADEMY & EXAMS",
        "AI INSTRUCTOR"
    ])

# -----------------------------------------------------------------------------
# 1. COMMAND DECK (Real-time & Social)
# -----------------------------------------------------------------------------
if nav == "COMMAND DECK":
    st.markdown("# 🛰️ GLOBAL COMMAND CENTER")
    
    # LIVE DATA ROW
    iss_lat, iss_lon = get_iss_location()
    sw = get_space_weather()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ISS LAT", f"{iss_lat:.4f}°", "LIVE")
    c2.metric("ISS LON", f"{iss_lon:.4f}°", "LIVE")
    c3.metric("SOLAR WIND", f"{sw['solar_wind']} km/s", f"Kp: {sw['kp_index']}")
    c4.metric("NEXT LAUNCH", "T-04:22:10", "Falcon 9 (Starlink)")
    
    # MAP
    st.markdown("### 🌍 ACTIVE TRACKING")
    map_data = pd.DataFrame({'lat': [iss_lat], 'lon': [iss_lon], 'type': ['ISS']})
    fig_map = px.scatter_geo(map_data, lat='lat', lon='lon', projection="orthographic")
    fig_map.update_geos(
        showland=True, landcolor="#111", showocean=True, oceancolor="#050505",
        showcountries=True, countrycolor="#333"
    )
    fig_map.update_traces(marker=dict(size=15, color="#00ff41", symbol="cross"))
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        paper_bgcolor="#000",
        height=400
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # DAILY CHALLENGE
    st.markdown("### 🎯 DAILY MISSION")
    with st.expander("MISSION: GEOSTATIONARY INJECTION", expanded=True):
        st.write("Calculate the Delta-V required to transfer from a 300km Parking Orbit to GEO (35,786km).")
        ans = st.number_input("Your Answer (km/s)", 0.0, 10.0, step=0.1)
        if st.button("SUBMIT SOLUTION"):
            # Hohmann math check
            if 3.8 <= ans <= 4.0:
                st.balloons()
                add_xp(100, "Daily Challenge Complete")
                st.success("CORRECT! Orbit synchronized.")
            else:
                st.error("INCORRECT. Check your Vis-Viva equation.")

# -----------------------------------------------------------------------------
# 2. SIM: ORBITAL (3D & Planning)
# -----------------------------------------------------------------------------
elif nav == "SIM: ORBITAL":
    st.markdown("# 🪐 ORBITAL DYNAMICS LAB")
    
    tabs = st.tabs(["3D VISUALIZER", "MISSION PLANNER", "SATELLITE DESIGN"])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("#### ORBIT PARAMS")
            alt = st.slider("Altitude (km)", 200, 36000, 400)
            inc = st.slider("Inclination (deg)", 0, 98, 51)
            ecc = st.slider("Eccentricity", 0.0, 0.9, 0.0)
            
            lats, lons, x, y, z, period = PhysicsCore.propagate_orbit(alt, inc, ecc)
            
            st.metric("ORBITAL PERIOD", f"{period/60:.1f} min")
            st.metric("VELOCITY", f"{math.sqrt(398600/(6371+alt)):.2f} km/s")
            
        with c2:
            # DUAL PLOT: GROUND TRACK + 3D
            fig = make_subplots(
                rows=1, cols=2, 
                specs=[[{'type': 'scattergeo'}, {'type': 'scatter3d'}]],
                column_widths=[0.4, 0.6],
                subplot_titles=("GROUND TRACK", "3D TRAJECTORY")
            )
            
            # Ground Track
            fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode='lines', line=dict(width=2, color='#00ff41')), row=1, col=1)
            fig.update_geos(projection_type="natural earth", showland=True, landcolor="#222", oceancolor="#111", row=1, col=1)
            
            # 3D Orbit
            # Earth Sphere
            phi = np.linspace(0, 2*np.pi, 20)
            theta = np.linspace(0, np.pi, 20)
            phi, theta = np.meshgrid(phi, theta)
            xe = 6371 * np.cos(phi) * np.sin(theta)
            ye = 6371 * np.sin(phi) * np.sin(theta)
            ze = 6371 * np.cos(theta)
            
            fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale='Electric', showscale=False, opacity=0.3), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='#00ff41', width=4)), row=1, col=2)
            
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with tabs[1]:
        st.markdown("#### 🚀 DELTA-V BUDGETER")
        # Multi-stage calculator
        stages = st.number_input("Number of Stages", 1, 3, 2)
        total_dv = 0
        
        for i in range(stages):
            c1, c2, c3 = st.columns(3)
            with c1: isp = st.number_input(f"Stage {i+1} ISP (s)", 200, 450, 300, key=f"isp_{i}")
            with c2: m0 = st.number_input(f"Stage {i+1} Wet Mass (kg)", 1000, 1000000, 50000, key=f"m0_{i}")
            with c3: mf = st.number_input(f"Stage {i+1} Dry Mass (kg)", 100, 100000, 5000, key=f"mf_{i}")
            
            if m0 > mf:
                dv = isp * 9.81 * math.log(m0/mf)
                st.info(f"Stage {i+1} Δv: {dv:.0f} m/s")
                total_dv += dv
            else:
                st.error(f"Stage {i+1} Invalid Mass")
                
        st.metric("TOTAL SYSTEM DELTA-V", f"{total_dv:.0f} m/s")
        
        # Mission Checker
        missions = {"LEO": 9400, "GTO": 11800, "Moon Land": 15000, "Mars Transfer": 13000}
        st.markdown("#### CAPABILITY CHECK")
        for m, req in missions.items():
            status = "✅" if total_dv > req else "❌"
            st.write(f"{status} **{m}** (Req: {req} m/s)")

# -----------------------------------------------------------------------------
# 3. WEAPONS & BALLISTICS
# -----------------------------------------------------------------------------
elif nav == "WEAPONS & BALLISTICS":
    st.markdown("# 💣 BALLISTICS RANGE")
    
    c1, c2, c3 = st.columns(3)
    v0 = c1.slider("Muzzle Velocity (m/s)", 300, 1500, 800)
    angle = c2.slider("Elevation Angle (deg)", 0, 90, 45)
    mass = c3.number_input("Projectile Mass (kg)", 0.1, 100.0, 10.0)
    
    c4, c5 = st.columns(2)
    drag_c = c4.slider("Drag Coeff (Cd)", 0.1, 1.0, 0.3)
    area = c5.slider("Cross Section (m^2)", 0.01, 0.5, 0.05)
    
    # Calculate
    tx, ty = PhysicsCore.ballistics_trajectory(v0, angle, drag_c, area, mass)
    
    # Analyze
    max_h = max(ty)
    range_x = max(tx)
    flight_time = len(tx) * 0.1
    
    # Visualize
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tx, y=ty, fill='tozeroy', line=dict(color='#ff5555'), name='Trajectory'))
    
    # Vacuum comparison
    vac_range = (v0**2 * math.sin(math.radians(2*angle))) / 9.81
    vac_x = np.linspace(0, vac_range, 100)
    vac_y = x = np.tan(math.radians(angle))*vac_x - (9.81/(2*v0**2*math.cos(math.radians(angle))**2))*vac_x**2
    fig.add_trace(go.Scatter(x=vac_x, y=vac_y, line=dict(dash='dot', color='#555'), name='Vacuum (No Drag)'))
    
    fig.update_layout(
        title="IMPACT ANALYSIS", 
        xaxis_title="Distance (m)", 
        yaxis_title="Altitude (m)", 
        template="plotly_dark",
        yaxis_scaleanchor="x"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAX RANGE", f"{range_x:.0f} m", f"Vacuum: {vac_range:.0f} m")
    col2.metric("APOGEE", f"{max_h:.0f} m")
    col3.metric("FLIGHT TIME", f"{flight_time:.1f} s")

# -----------------------------------------------------------------------------
# 4. ENGINEERING TOOLS
# -----------------------------------------------------------------------------
elif nav == "ENGINEERING TOOLS":
    st.markdown("# 🛠️ ENGINEER'S WORKBENCH")
    
    t1, t2 = st.tabs(["UNIT CONVERTER", "TRADE STUDY"])
    
    with t1:
        st.markdown("### SMART CONVERTER")
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Value", value=1.0)
        cat = c2.selectbox("Category", ["Speed", "Distance", "Mass"])
        
        if cat == "Speed":
            res_mach = val / 340.3
            res_kmh = val * 3.6
            c3.write(f"➝ **{res_mach:.2f} Mach** (SL)")
            c3.write(f"➝ **{res_kmh:.2f} km/h**")
        elif cat == "Distance":
            res_nm = val * 0.539957
            res_ft = val * 3280.84
            c3.write(f"➝ **{res_nm:.2f} nm**")
            c3.write(f"➝ **{res_ft:.2f} ft**")
            
    with t2:
        st.markdown("### DECISION MATRIX")
        st.caption("Compare design options (1-5 Scale)")
        
        opts = st.text_input("Options (comma sep)", "Design A, Design B").split(',')
        criteria = st.text_input("Criteria (comma sep)", "Cost, Weight, Performance").split(',')
        
        data = {}
        for c in criteria:
            data[c] = [random.randint(1,5) for _ in opts]
            
        df = pd.DataFrame(data, index=opts)
        df['SCORE'] = df.sum(axis=1)
        
        st.dataframe(df.style.highlight_max(axis=0, color='#004400'))
        st.bar_chart(df['SCORE'])

# -----------------------------------------------------------------------------
# 5. ACADEMY & EXAMS
# -----------------------------------------------------------------------------
elif nav == "ACADEMY & EXAMS":
    st.markdown("# 🎓 PILOT ACADEMY")
    
    mode = st.radio("SELECT MODE", ["LIBRARY", "AFCAT/NDA MOCK EXAM"], horizontal=True)
    
    if mode == "LIBRARY":
        st.markdown("""
        ### 📚 TECHNICAL GLOSSARY
        - **Apogee:** The point in an orbit furthest from the body being orbited.
        - **Beta Angle:** The angle between the orbital plane and the vector to the sun.
        - **Coordinated Turn:** A turn where the aircraft slips nor skids (ball centered).
        - **Delta-V:** The change in velocity required to perform a maneuver.
        - **Specific Impulse (Isp):** Efficiency of a rocket engine (seconds).
        """)
        
        st.info("💡 PRO TIP: High Aspect Ratio wings (gliders) produce less induced drag but roll slower.")
        
    else:
        st.markdown("### 📝 MOCK TEST (Topic: Orbital Mechanics)")
        
        questions = [
            {"q": "What happens to orbital velocity as altitude increases?", "opts": ["Increases", "Decreases", "Constant"], "a": "Decreases"},
            {"q": "Which law states 'Equal areas in equal times'?", "opts": ["Newton 1", "Kepler 2", "Kepler 3"], "a": "Kepler 2"},
            {"q": "What is the escape velocity of Earth approx?", "opts": ["7.8 km/s", "11.2 km/s", "9.8 km/s"], "a": "11.2 km/s"},
            {"q": "Where is kinetic energy highest in an elliptical orbit?", "opts": ["Apogee", "Perigee", "Nodes"], "a": "Perigee"},
            {"q": "A Hohmann transfer is:", "opts": ["Fastest", "Most Fuel Efficient", "Linear"], "a": "Most Fuel Efficient"}
        ]
        
        score = 0
        with st.form("exam_form"):
            for i, q in enumerate(questions):
                ans = st.radio(f"{i+1}. {q['q']}", q['opts'], key=f"q_{i}")
                if ans == q['a']: score += 1
            
            sub = st.form_submit_button("SUBMIT EXAM")
            if sub:
                pct = score / len(questions) * 100
                st.progress(pct/100)
                if pct >= 80:
                    st.success(f"PASSED! Score: {score}/{len(questions)}")
                    add_xp(200, "Exam Passed")
                else:
                    st.error(f"FAILED. Score: {score}/{len(questions)}")

# -----------------------------------------------------------------------------
# 6. AI INSTRUCTOR
# -----------------------------------------------------------------------------
elif nav == "AI INSTRUCTOR":
    st.markdown("# 🤖 FLIGHT INSTRUCTOR AI")
    
    if not AI_ENABLED:
        st.warning("⚠️ GROQ API KEY MISSING. AI OFFLINE.")
    else:
        # Chat History
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "assistant"
            st.chat_message(role).write(msg["content"])
            
        if prompt := st.chat_input("Ask about physics, tactics, or exam prep..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Context Building
            context = f"""
            User Rank: {st.session_state.user_profile['rank']}
            Current XP: {st.session_state.user_profile['xp']}
            Role: Expert Air Force Instructor & Physicist.
            Tone: Strict but encouraging. Use LaTeX for math.
            """
            
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": context},
                        *st.session_state.messages
                    ],
                    stream=True
                )
                with st.chat_message("assistant"):
                    response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"AI Error: {e}")

# --- FOOTER ---
st.markdown("---")
c1, c2 = st.columns([3, 1])
c1.caption(f"MACH-X OMNI | v3.1 | USER: {st.session_state.user_profile['rank']}")
if c2.button("💾 SAVE & EXPORT"):
    st.toast("Profile Saved to Local Storage (Simulated)")
