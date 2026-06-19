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

# cache lines for 1 hour so we dont hit the API on every page load
@st.cache_data(ttl=3600)
def load_lines():
    return parse_lines(fetch_strikeout_lines())

lines = load_lines()

# cache each pitcher's stats so we only pull Statcast data once per day
@st.cache_data(ttl=86400)
def get_pitcher_projection(pitcher_name):
    try:
        from data.fetch_statcast import get_player_id
        player_id = get_player_id(pitcher_name)
        df = fetch_pitcher_statcast(player_id, '2025-03-01', '2025-12-31')
        df = aggregate_to_starts(df)
        df = rolling_features(df)
        df = df.dropna()
        if len(df) == 0:
            return None
        last = df.iloc[-1]
        return predict_strikeouts(
            rolling_k=last['rolling_k'],
            rolling_swstr=last['rolling_swstr'],
            rolling_velocity=last['rolling_velocity'],
            rolling_pitches=last['rolling_pitches']
        )
    except:
        return None

if search:
    lines = [line for line in lines if search.lower() in line['pitcher'].lower()]

# display a card for each pitcher with their line and projection
for line in lines:
    with st.container(border=True):
        with st.spinner(f"Loading {line['pitcher']}..."):
            predicted_ks = get_pitcher_projection(line['pitcher'])
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {line['pitcher']}")
            st.caption(line['game'])

        with col2:
            st.metric(label="Line", value=line['line'])
            if predicted_ks is not None:
                st.metric(label="Projection", value=f"{predicted_ks:.1f} K")
            else:
                st.metric(label="Projection", value="N/A")

        if predicted_ks is not None:
            diff = predicted_ks - line['line']
            if diff > 0:
                st.success(f"↑ LEAN OVER — {diff:.1f} above the line")
            elif diff < 0:
                st.error(f"↓ LEAN UNDER — {abs(diff):.1f} below the line")
            else:
                st.warning("⚖ TOO CLOSE TO CALL")
        else:
            st.warning("No 2025 data available for this pitcher")