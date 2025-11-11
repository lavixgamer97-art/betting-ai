import streamlit as st
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Dynamisk path oberoende av dator
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "football_prematch_model.pkl"
FEATURES_PATH = BASE_DIR / "pl_features.csv"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    # Bygg upp klass-index så vi läser predict_proba rätt oavsett ordning
    class_to_index = {cls: i for i, cls in enumerate(model.classes_)}
    return model, class_to_index

@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES_PATH)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # Grundkoll
    needed = [
        'Date', 'HomeTeam', 'AwayTeam',
        'home_gf_avg_last5', 'home_ga_avg_last5', 'home_pts_avg_last5',
        'away_gf_avg_last5', 'away_ga_avg_last5', 'away_pts_avg_last5'
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Saknar kolumner i pl_features.csv: {missing}")
        st.stop()

    return df

def build_team_form(df):
    """Hämta senaste formvärden (gf/ga/pts last5) för varje lag."""
    teams = set(df['HomeTeam']).union(set(df['AwayTeam']))
    team_form = {}

    for team in teams:
        h = df[df['HomeTeam'] == team].sort_values('Date')
        a = df[df['AwayTeam'] == team].sort_values('Date')

        candidates = []

        if not h.empty:
            last_h = h.iloc[-1]
            candidates.append((
                last_h['Date'],
                last_h['home_gf_avg_last5'],
                last_h['home_ga_avg_last5'],
                last_h['home_pts_avg_last5'],
            ))

        if not a.empty:
            last_a = a.iloc[-1]
            candidates.append((
                last_a['Date'],
                last_a['away_gf_avg_last5'],
                last_a['away_ga_avg_last5'],
                last_a['away_pts_avg_last5'],
            ))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, gf, ga, pts = candidates[-1]
            team_form[team] = {
                'gf': float(gf),
                'ga': float(ga),
                'pts': float(pts),
            }
        else:
            team_form[team] = {'gf': 0.0, 'ga': 0.0, 'pts': 0.0}

    return team_form

# =========================
# App logic
# =========================

st.set_page_config(
    page_title="Match Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ AI Match Predictor")
st.write(
    "Mata in omgångens matcher så räknar modellen ut 1/X/2-sannolikheter "
    "baserat på lagens form (mål & poäng senaste 5)."
)

model, class_to_index = load_model()
df_features = load_features()
team_form = build_team_form(df_features)

teams_sorted = sorted(team_form.keys())

st.subheader("1. Välj matcher")

num_matches = st.slider("Antal matcher", min_value=1, max_value=20, value=5)

matches = []
cols = st.columns(3)
cols[0].markdown("**Hemma**")
cols[1].markdown("**Borta**")
cols[2].markdown("&nbsp;")

for i in range(num_matches):
    c1, c2, _ = st.columns([3, 3, 1])
    home = c1.selectbox(
        f"Hemmalag {i+1}",
        options=["(välj lag)"] + teams_sorted,
        key=f"home_{i}"
    )
    away = c2.selectbox(
        f"Bortalag {i+1}",
        options=["(välj lag)"] + teams_sorted,
        key=f"away_{i}"
    )

    if home != "(välj lag)" and away != "(välj lag)" and home != away:
        matches.append((home, away))

st.write("---")

if st.button("🔮 Beräkna prediktioner", type="primary"):
    if not matches:
        st.warning("Lägg in minst en giltig match (olika lag) först.")
    else:
        rows = []
        for home, away in matches:
            if home not in team_form or away not in team_form:
                st.warning(f"Saknar formdata för {home} eller {away}, hoppar över.")
                continue

            hf = team_form[home]
            af = team_form[away]

            X = pd.DataFrame([[
                hf['gf'], hf['ga'], hf['pts'],
                af['gf'], af['ga'], af['pts'],
            ]], columns=[
                'home_gf_avg_last5',
                'home_ga_avg_last5',
                'home_pts_avg_last5',
                'away_gf_avg_last5',
                'away_ga_avg_last5',
                'away_pts_avg_last5',
            ])

            proba = model.predict_proba(X)[0]
            pred_class = model.predict(X)[0]

            # Plocka rätt sannolikhet per klass oavsett ordning
            p_X = float(proba[class_to_index[0]]) if 0 in class_to_index else 0.0
            p_1 = float(proba[class_to_index[1]]) if 1 in class_to_index else 0.0
            p_2 = float(proba[class_to_index[2]]) if 2 in class_to_index else 0.0

            # label_map enligt vår encoding
            label_map = {1: "1 (hemmavinst)", 0: "X (oavgjort)", 2: "2 (bortavinst)"}
            best_label = label_map.get(pred_class, "?")
            best_p = max(p_1, p_X, p_2)

            rows.append({
                "Match": f"{home} - {away}",
                "AI Tips": best_label,
                "P(1)": round(p_1, 3),
                "P(X)": round(p_X, 3),
                "P(2)": round(p_2, 3),
                "Säkerhet": round(best_p, 3),
            })

        if not rows:
            st.error("Inga matcher kunde beräknas. Kontrollera lagvalen.")
        else:
            # Sortera efter säkerhet
            rows_sorted = sorted(rows, key=lambda r: r["Säkerhet"], reverse=True)

            st.subheader("2. Resultat – rankade matcher")
            st.caption("Högst 'Säkerhet' = modellen är mest trygg i sin prediktion (inte samma som gratis pengar 😉).")

            st.dataframe(pd.DataFrame(rows_sorted), use_container_width=True)
