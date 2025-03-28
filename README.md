
# HNTR AI Financial Advisor Scoring App

This is a simple Streamlit app that calculates financial advisor scores based on multiple factors such as GDC, AUM, competitor site visits, and event attendance. It performs clustering to group advisors and provides downloadable CSV data of the processed results.

## How to Use

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the app using `streamlit run hntr_ai_app.py`.
3. Upload a CSV file with columns such as `AUM`, `GDC`, `competitor_site_visits`, `event_attendance`, and `name`.
4. The app will process the data, calculate BLIX, Fit, and Priority scores, perform clustering, and display results.
5. You can download the processed data as a CSV file.
