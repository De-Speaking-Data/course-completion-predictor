import streamlit as st, joblib, json, pandas as pd

st.set_page_config(page_title="Course Completion Predictor", page_icon="🎓")

@st.cache_resource
def load():
    return joblib.load("model.joblib"), json.load(open("meta.json"))

model, meta = load()
cols, thr, cats = meta["columns"], meta["threshold"], meta["cats"]

st.title("🎓 HarvardX Course Completion Predictor")
st.write("Estimates whether a student will earn a certificate from their engagement. Demo model trained on HarvardX 2012–13 data.")

with st.form("f"):
    c1, c2 = st.columns(2)
    course = c1.selectbox("Course", cats["course_id"])
    country = c2.selectbox("Country", cats["final_cc_cname_DI"])
    gender = c1.selectbox("Gender", cats["gender"])
    loe = c2.selectbox("Level of education", cats["LoE_DI"])
    yob = c1.number_input("Year of birth", 1940, 2013, 1987)
    duration = c2.number_input("Active duration (days)", 0.0, 400.0, 0.0)
    nevents = c1.number_input("Number of events", 0, 20000, 0)
    ndays = c2.number_input("Days active", 0, 400, 0)
    nvideo = c1.number_input("Videos played", 0, 6000, 0)
    nchap = c2.number_input("Chapters accessed", 0, 40, 0)
    nforum = c1.number_input("Forum posts", 0, 500, 0)
    viewed = c2.checkbox("Viewed the course")
    incomplete = c2.checkbox("Incomplete flag")
    go = st.form_submit_button("Predict")

if go:
    row = dict.fromkeys(cols, 0)
    row.update(registered=1, viewed=int(viewed), YoB=yob, age=2013 - yob, nevents=nevents,
               ndays_act=ndays, nplay_video=nvideo, nchapters=nchap, nforum_posts=nforum,
               incomplete_flag=int(incomplete), active_duration_days=duration)
    for key in [f"course_id_{course}", f"final_cc_cname_DI_{country}", f"gender_{gender}", f"LoE_DI_{loe}"]:
        if key in row:
            row[key] = 1
    p = float(model.predict_proba(pd.DataFrame([row])[cols])[0, 1])
    st.metric("Probability of passing", f"{p:.0%}")
    st.progress(p)
    if p >= thr:
        st.success(f"LIKELY TO PASS (score {p:.2f}, cut-off {thr})")
    else:
        st.error(f"UNLIKELY TO PASS (score {p:.2f}, cut-off {thr})")
