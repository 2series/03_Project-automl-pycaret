import os
from operator import index

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Import profiling libraries
import pandas_profiling
import plotly.express as px
import streamlit as st
from pycaret.classification import (
    compare_models,
    load_model,
    plot_model,
    pull,
    save_model,
    setup,
)

# Import ML libraries
from pycaret.regression import compare_models as compare_regression_models
from pycaret.regression import load_model as load_regression_model
from pycaret.regression import pull as regression_pull
from pycaret.regression import save_model as save_regression_model
from pycaret.regression import setup as regression_setup
from streamlit_pandas_profiling import st_profile_report

with st.sidebar:
    st.image("porcelain profile.png")
    st.title("AutoML App")
    choice = st.radio(
        "Select a task", ["Upload", "Profiling", "Machine Learning", "Download"]
    )
    st.info("This application helps you quickly build and explore your data")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload a CSV file")
    file = st.file_uploader("Upload a CSV file", type="csv")
    if file:
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
        df.to_csv("sourcedata.csv", index=False)


if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    export = profile_report.to_html()
    st.download_button(
        label="Download Full Report", data=export, file_name="report.html"
    )


if choice == "Machine Learning":
    st.title("Machine Learning magic at your fingertips")
    target = st.selectbox("Select a target column", df.columns)
    task = st.radio(
        "Select task type",
        ["Regression", "Binary Classification", "Multiclass Classification"],
    )
    if st.button("Run Modeling"):
        if task == "Regression":
            regression_setup(df, target=target, verbose=False)
            setup_df = regression_pull()
            st.info("This is the Regression ML experiment settings")
            st.dataframe(setup_df)
            best_model = compare_regression_models()
            compare_df = regression_pull()
            st.info("This is the ML model")
            st.dataframe(compare_df)
            save_regression_model(best_model, "best_model")

        elif task == "Binary Classification":
            setup(df, target=target, verbose=False, html=False)
            setup_df = pull()
            setup_df.loc[2, "Target type"] = "Classification"  # Update target type
            st.info("This is the Binary Classification ML experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML model")
            st.dataframe(compare_df)
            save_model(best_model, "best_model")

        elif task == "Multiclass Classification":
            setup(df, target=target, verbose=False, html=False)
            setup_df = pull()
            setup_df.loc[
                2, "Target type"
            ] = "Multiclass Classification"  # Update target type
            st.info("This is the Multiclass Classification ML experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML model")
            st.dataframe(compare_df)
            save_model(best_model, "best_model")

if choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download Model", f, file_name="best_model.pkl")
