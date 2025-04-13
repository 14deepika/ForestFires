# frontend.py
import streamlit as st
import requests
import numpy as np
import plotly.express as px

backend_url = "http://localhost:5000"

st.title("Forest Fire Dynamics: CA + ML")

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose a mode:", ["Predict Single Cell", "Run CA Simulation"])

if mode == "Predict Single Cell":
    st.header("Predict Ignition Risk for a Single Cell")

    X = st.number_input("X (Row Index)", value=4, step=1)
    Y = st.number_input("Y (Column Index)", value=5, step=1)
    month = st.number_input("Month (1=Jan, 12=Dec)", min_value=1, max_value=12, value=8)
    day = st.number_input("Day (1=Mon, 7=Sun)", min_value=1, max_value=7, value=1)
    FFMC = st.number_input("FFMC", value=90.0)
    DMC = st.number_input("DMC", value=35.0)
    DC = st.number_input("DC", value=100.0)
    ISI = st.number_input("ISI", value=5.0)
    temp = st.number_input("Temperature (°C)", value=20.0)
    RH = st.number_input("Relative Humidity (%)", value=40)
    wind = st.number_input("Wind (km/h)", value=3.0)
    rain = st.number_input("Rain (mm)", value=0.0)

    if st.button("Predict Fire Risk"):
        data = {
            "X": X, "Y": Y, "month": month, "day": day,
            "FFMC": FFMC, "DMC": DMC, "DC": DC, "ISI": ISI,
            "temp": temp, "RH": RH, "wind": wind, "rain": rain
        }
        try:
            response = requests.post(f"{backend_url}/predict", json=data)
            if response.status_code == 200:
                result = response.json()
                fire_prob = result["fire_risk_prob"]
                fire_class = result["fire_risk_class"]

                st.write(f"**Predicted Class:** {fire_class}")
                st.write(f"**Probability:** {fire_prob:.2f}")

                if fire_prob >= 0.5:
                    st.error("**ALERT: High Fire Risk (Unsafe)**")
                else:
                    st.success("**Safe: Low Fire Risk**")
            else:
                st.error("Prediction failed. Check backend logs.")
        except Exception as e:
            st.error(f"Error: {e}")

elif mode == "Run CA Simulation":
    st.header("Cellular Automata Simulation")

    rows = st.number_input("Grid Rows", value=10, step=1)
    cols = st.number_input("Grid Columns", value=10, step=1)
    steps = st.number_input("Number of Steps", value=5, step=1)

    st.write("Environment Data:")
    month = st.number_input("Month (1=Jan, 12=Dec)", min_value=1, max_value=12, value=8)
    day = st.number_input("Day (1=Mon, 7=Sun)", min_value=1, max_value=7, value=1)
    FFMC = st.number_input("FFMC", value=90.0)
    DMC = st.number_input("DMC", value=35.0)
    DC = st.number_input("DC", value=100.0)
    ISI = st.number_input("ISI", value=5.0)
    temp = st.number_input("Temperature (°C)", value=20.0)
    RH = st.number_input("Relative Humidity (%)", value=40)
    wind = st.number_input("Wind (km/h)", value=3.0)
    rain = st.number_input("Rain (mm)", value=0.0)

    if st.button("Run Simulation"):
        env_data = {
            "month": month,
            "day": day,
            "FFMC": FFMC,
            "DMC": DMC,
            "DC": DC,
            "ISI": ISI,
            "temp": temp,
            "RH": RH,
            "wind": wind,
            "rain": rain
        }
        payload = {
            "rows": rows,
            "cols": cols,
            "steps": steps,
            "env_data": env_data
        }
        try:
            response = requests.post(f"{backend_url}/simulate", json=payload)
            if response.status_code == 200:
                result = response.json()
                final_grid = np.array(result["final_grid"])

                st.write("### Final CA Grid (0=unburned, 1=burning, 2=burned)")

                color_scale = [
                    [0.0, "green"],  # unburned
                    [0.5, "red"],    # burning
                    [1.0, "black"]   # burned
                ]
                fig = px.imshow(
                    final_grid,
                    color_continuous_scale=color_scale,
                    zmin=0,
                    zmax=2,
                    aspect="equal",
                    labels=dict(x="Column", y="Row", color="State")
                )
                st.plotly_chart(fig)
            else:
                st.error("Simulation failed. Check backend logs.")
        except Exception as e:
            st.error(f"Error: {e}")
