import streamlit as st
import sys
sys.path.append('.')
from data.fetch_statcast import fetch_pitcher_statcast, aggregate_to_starts
from features.engineer import rolling_features
from model.predict import predict_strikeouts
from odds.fetch_lines import parse_lines 
from odds.fetch_lines import fetch_strikeout_lines

st.set_page_config(page_title = "Sports Predcitor", layout = "wide")
st.title("Gain an Edge")
st.subheader("MLB Pitcher Strikout Projections")
search = st.text_input("Search Pitcher...")
lines = parse_lines(fetch_strikeout_lines())

if search:
    lines = [line for line in lines if search.lower() in line['pitcher'].lower()]

# display a card for each pitcher with their line and projection
for line in lines:
    with st.container(border=True):
        # placeholder rolling stats until we wire up live Statcast data
        predicted_ks = 6.2  # placeholder
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {line['pitcher']}")
            st.caption(line['game'])
        
        with col2:
            st.metric(label="Line", value=line['line'])
            st.metric(label="Projection", value=f"{predicted_ks} K")

        #calculate differnece and display lean
        diff = predicted_ks - line['line']
        if diff>0:
            st.success(f"↑ LEAN OVER — {diff:.1f} above the line")
        elif diff < 0:
            st.error(f"↓ LEAN UNDER — {abs(diff):.1f} below the line")
        else:
            st.warning("⚖ TOO CLOSE TO CALL")


