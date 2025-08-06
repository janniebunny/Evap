import streamlit as st
import numpy as np
import pandas as pd

# ---- BPE and Water BP lookup functions ----
def water_bp_kpa(temp_c):
    """Return approximate absolute pressure [kPa] at given boiling point [C] for water. Uses Antoine eqn."""
    A, B, C = 8.07131, 1730.63, 233.426
    p_mmHg = 10**(A - B / (temp_c + C))
    p_kPa = p_mmHg * 0.133322
    return p_kPa

def water_bp_temp(pressure_kpa):
    table = [
        (0.6, 0), (1.2, 10), (2.3, 20), (4.2, 30), (7.4, 40),
        (12.3, 50), (19.9, 60), (31.8, 70), (47.4, 80), (67.0, 90),
        (101.3, 100), (200, 120), (300, 134), (400, 144), (500, 151)
    ]
    ps, ts = zip(*table)
    return np.interp(pressure_kpa, ps, ts)

def bpe_nacl(conc_pct):
    if conc_pct <= 0:
        return 0
    return np.interp(conc_pct, [0, 1, 5, 10, 15], [0, 0.3, 1.5, 4, 7])

# ---- Calculation Logic ----
def evaporator_calc(
    n_effects, feed_flow_m3h, feed_dm_pct, feed_tss_pct, feed_temp_c, prod_dm_pct, steam_press_barg
):
    results = []
    density = 1.0  # kg/L
    steam_press_abs = steam_press_barg + 1  # [bar abs]
    steam_press_kpa = steam_press_abs * 100
    steam_temp_c = water_bp_temp(steam_press_kpa)

    feed_flow_kgph = feed_flow_m3h * 1000
    feed_dm_kgph = feed_dm_pct/100 * feed_flow_kgph
    feed_tss_kgph = feed_tss_pct/100 * feed_flow_kgph
    feed_tds_kgph = feed_dm_kgph - feed_tss_kgph
    prod_flow_kgph = feed_dm_kgph / (prod_dm_pct/100) if prod_dm_pct > 0 else 0
    evap_kgph = feed_flow_kgph - prod_flow_kgph
    prod_flow_m3h = prod_flow_kgph / 1000

    dm_concs = np.linspace(feed_dm_pct, prod_dm_pct, n_effects+1)
    bpes = [bpe_nacl(max(0.01, dm_concs[i] - feed_tss_pct)) for i in range(1, n_effects+1)]
    eff_press_kpa = [None]*n_effects
    bp_temp_c = [None]*n_effects
    for i in range(n_effects-1, -1, -1):
        if i == n_effects-1:
            eff_press_kpa[i] = 25
            bp_temp_c[i] = water_bp_temp(25) + bpes[i]
        else:
            dt = (steam_temp_c - (water_bp_temp(25)+bpes[-1]))/n_effects
            bp_temp_c[i] = bp_temp_c[i+1] + dt
            eff_press_kpa[i] = water_bp_kpa(bp_temp_c[i] - bpes[i])
    lmtds = []
    for i in range(n_effects):
        if i == 0:
            hot = steam_temp_c
        else:
            hot = bp_temp_c[i-1]
        cold = bp_temp_c[i]
        dT1 = hot - cold
        if dT1 <= 0.1: dT1 = 0.1
        lmtds.append(round(dT1,1))

    # ---- Per stage mass balance with validation ----
    stage_conc_flow = [feed_flow_kgph]
    stage_conc_m3h = [feed_flow_kgph / 1000]
    stage_dm = [feed_dm_kgph]
    stage_vap_kgph = []
    stage_vap_m3h = []
    error_flag = False
    error_msg = ""
    for i in range(n_effects):
        prev_conc_kgph = stage_conc_flow[-1]
        prev_dm_kgph = stage_dm[-1]
        this_dm_pct = dm_concs[i+1] / 100
        if this_dm_pct <= 0:
            error_flag = True
            error_msg = f"Error: Dry matter in effect {i+1} is zero or negative."
            next_conc_kgph = 0
            vap_kgph = 0
        elif prev_dm_kgph == 0:
            error_flag = True
            error_msg = f"Error: Zero dry matter in effect {i+1}."
            next_conc_kgph = 0
            vap_kgph = 0
        else:
            next_conc_kgph = prev_dm_kgph / this_dm_pct
            vap_kgph = prev_conc_kgph - next_conc_kgph
        stage_conc_flow.append(next_conc_kgph)
        stage_dm.append(prev_dm_kgph)
        stage_vap_kgph.append(vap_kgph)
        stage_conc_m3h.append(next_conc_kgph / 1000)
        stage_vap_m3h.append(vap_kgph / 1000)

    if prod_dm_pct < 12:
        steam_econ = 2.3
    elif prod_dm_pct < 18:
        steam_econ = 2.1
    else:
        steam_econ = 1.8
    steam_needed_kgph = evap_kgph / steam_econ if steam_econ > 0 else 0
    latent_heat = 2300  # kJ/kg
    total_thermal_kW = evap_kgph * latent_heat / 3600
    steam_power_kW = steam_needed_kgph * latent_heat / 3600
    lv_condenser = 2350  # kJ/kg at ~65–70°C
    condenser_kW = evap_kgph * lv_condenser / 3600  # kW

    for i in range(n_effects):
        results.append({
            'Effect': i+1,
            'Abs Pressure (kPa)': round(eff_press_kpa[i],1),
            'Boiling Pt (°C)': round(bp_temp_c[i],1),
            'BPE (°C)': round(bpes[i],2),
            'LMTD (°C)': lmtds[i],
            'Vapour Flow (kg/h)': int(stage_vap_kgph[i]),
            'Vapour Flow (m³/h)': round(stage_vap_m3h[i],2),
            'Concentrate Flow (kg/h)': int(stage_conc_flow[i+1]),
            'Concentrate Flow (m³/h)': round(stage_conc_m3h[i+1],2),
        })
    output = {
        'Feed Flow (m³/h)': feed_flow_m3h,
        'Feed DM (%)': feed_dm_pct,
        'Feed TSS (%)': feed_tss_pct,
        'Feed TDS (%)': round(feed_tds_kgph/feed_flow_kgph*100,2) if feed_flow_kgph > 0 else 0,
        'Product Flow (m³/h)': round(prod_flow_m3h,2),
        'Product DM (%)': prod_dm_pct,
        'Water Evaporated (kg/h)': int(evap_kgph),
        'Steam Needed (kg/h)': int(steam_needed_kgph),
        'Steam Power (kW)': int(steam_power_kW),
        'Steam Economy': round(steam_econ,2),
        'Effects': results,
        'Total Thermal Load (kW)': int(total_thermal_kW),
        'Steam Temp (°C)': round(steam_temp_c,1),
        'Final Stage Condenser Load (kW)': int(condenser_kW),
        'Error': error_msg if error_flag else ""
    }
    return output

# ---- Streamlit UI ----
st.set_page_config(page_title="Multi-effect Evaporator Design Tool", layout="wide")

# --- Company logo top right ---
col1, col2 = st.columns([6,1])
with col1:
    st.title("Multi-effect Evaporator Design Calculator")
with col2:
    st.image("logo.png", width=160)  # Adjust width as needed

st.markdown("""
*All calculations based on standard thermodynamics and referenced correlations. Results are for indicative engineering only. BPE values use NaCl curve as basis. For other salts, contact admin.*
""")


col1, col2 = st.columns(2)
with col1:
    n_effects = st.slider("Number of Effects", 2, 4, 3)
    feed_flow_m3h = st.number_input("Feed Flow (m³/h)", 1.0, 500.0, 41.67)
    feed_dm_pct = st.number_input("Feed Dry Matter (%)", 0.1, 15.0, 2.38)
    feed_tss_pct = st.number_input("Feed TSS (%)", 0.0, 10.0, 1.5)
with col2:
    prod_dm_pct = st.number_input("Product Dry Matter (%)", 2.0, 50.0, 15.0)
    feed_temp_c = st.number_input("Feed Temperature (°C)", 0.0, 99.0, 25.0)
    steam_press_barg = st.number_input("Steam Pressure (bar(g))", 0.5, 10.0, 1.0)

if st.button("Calculate Evaporator Design"):
    try:
        output = evaporator_calc(
            n_effects, feed_flow_m3h, feed_dm_pct, feed_tss_pct, feed_temp_c, prod_dm_pct, steam_press_barg
        )
        if output['Error']:
            st.error(output['Error'])
        else:
            st.subheader("Mass & Energy Balance")
            st.write({k:v for k,v in output.items() if k not in ['Effects', 'Error']})
            st.subheader("Effect-wise Summary")
            st.dataframe(pd.DataFrame(output['Effects']))
            st.info(f"Steam temperature: {output['Steam Temp (°C)']}°C | Steam economy: {output['Steam Economy']}")
            st.info(f"Total thermal load: {output['Total Thermal Load (kW)']} kW | Steam-side: {output['Steam Power (kW)']} kW")
            st.info(f"Final condenser load (thermal, kW): {output['Final Stage Condenser Load (kW)']}")
    except Exception as e:
        st.error(f"Calculation error: {e}")

st.markdown("""
---
*References: Perry’s Chem Eng Handbook, IAPWS steam tables, GEA process data.*
""")
