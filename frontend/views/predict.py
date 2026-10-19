"""Make a Prediction: a form that calls POST /predict on the Champion model."""

from datetime import date, time

import streamlit as st

import api_client
from components.ui import page_header, show_error


page_header(
    "Make a Prediction",
    "Enter a taxi trip and the current Champion model predicts how long it "
    "will take. The request is logged and shows up on the Predictions page.",
)

# Well-known NYC TLC taxi zones (full list: 1-263)
ZONES = {
    "JFK Airport": 132,
    "LaGuardia Airport": 138,
    "Newark Airport": 1,
    "Times Sq / Theatre District": 230,
    "Midtown Center": 161,
    "Midtown East": 162,
    "Penn Station / Madison Sq West": 186,
    "Union Sq": 234,
    "East Village": 79,
    "Clinton East": 48,
    "Upper East Side South": 237,
    "Upper East Side North": 236,
}
OTHER = "Other (enter zone ID)"

PAYMENT_TYPES = {"Credit card": 1, "Cash": 2, "No charge": 3,
                 "Dispute": 4, "Unknown": 5, "Voided trip": 6}
RATE_CODES = {"Standard rate": 1, "JFK": 2, "Newark": 3,
              "Nassau or Westchester": 4, "Negotiated fare": 5, "Group ride": 6}
VENDORS = {"Creative Mobile Technologies": 1, "VeriFone / Curb": 2}


def zone_picker(label: str, default: str, key: str) -> int:
    names = list(ZONES) + [OTHER]
    choice = st.selectbox(label, names, index=names.index(default), key=key)
    if choice == OTHER:
        return int(st.number_input(f"{label} ID", 1, 263, 100, key=key + "_id"))
    return ZONES[choice]


with st.form("predict"):
    c1, c2 = st.columns(2)
    with c1:
        pickup_zone = zone_picker("Pickup zone", "Times Sq / Theatre District", "pu")
    with c2:
        dropoff_zone = zone_picker("Dropoff zone", "Midtown Center", "do")

    c1, c2, c3 = st.columns(3)
    distance = c1.number_input("Trip distance (miles)", 0.1, 200.0, 3.5, 0.1)
    pickup_date = c2.date_input("Pickup date", date(2026, 3, 2))
    pickup_time = c3.time_input("Pickup time", time(8, 30))

    c1, c2, c3 = st.columns(3)
    payment = c1.selectbox("Payment type", list(PAYMENT_TYPES))
    rate = c2.selectbox("Rate code", list(RATE_CODES))
    vendor = c3.selectbox("Vendor", list(VENDORS), index=1)

    submitted = st.form_submit_button("Predict trip duration", type="primary")

if submitted:
    payload = {
        "trip_distance": float(distance),
        "pickup_datetime": f"{pickup_date.isoformat()}T{pickup_time.strftime('%H:%M:%S')}",
        "pulocationid": pickup_zone,
        "dolocationid": dropoff_zone,
        "payment_type": PAYMENT_TYPES[payment],
        "vendorid": VENDORS[vendor],
        "ratecodeid": RATE_CODES[rate],
    }

    with st.spinner("Asking the Champion model..."):
        result = api_client.predict(payload)

    if not result.ok:
        show_error(result, "Prediction")
        st.stop()

    out = result.data
    minutes = out["prediction_minutes"]

    st.success(f"Estimated trip duration: **{minutes:.1f} minutes**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{minutes:.1f} min", f"{out['prediction_seconds']:.0f} s",
              delta_color="off")
    c2.metric("Model version", f"v{out['model_version']}")
    c3.metric("Latency", f"{out['latency_ms']:.0f} ms")
