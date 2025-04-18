import os, io, base64

# 1) Force Matplotlib into non‑GUI mode
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import calendar   # ← for month names

app = Flask(__name__)

# fixed list of your five data‑source categories
CSV_OPTIONS = [
    'All_Residential',
    'Condo',
    'Multi_Family',
    'Single_Family',
    'Townhouse',
    'Rental_pred'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/map")
def map_page():
    return render_template("map.html")

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    # On GET, just render the form with our CSV choices
    if request.method == 'GET':
        return render_template('graph.html', data_sources=CSV_OPTIONS)

    # POST → figure out which CSV to load
    selected_src = request.form.get('data_src')
    if selected_src not in CSV_OPTIONS:
        # invalid selection, fall back to first
        selected_src = CSV_OPTIONS[0]

    # build the path to static/<Category>.csv
    DATA_PATH = os.path.join(
        app.root_path,
        'static',
        f"{selected_src}.csv"
    )

    # load CSV & ensure Zip code is a string
    df = pd.read_csv(DATA_PATH)
    df['Zip code'] = df['Zip code'].astype(str)

    # grab raw input, split on commas
    raw = request.form.get('zip_codes', '')
    selected = [z.strip() for z in raw.split(',') if z.strip()]

    if not selected:
        return render_template('graph.html',
                               data_sources=CSV_OPTIONS,
                               error="Please enter at least one ZIP code.",
                               raw_input=raw,
                               current_src=selected_src)

    # build the figure
    fig, ax = plt.subplots(figsize=(6,4))
    missing = []

    for z in selected:
        sub = df[df['Zip code'] == z]
        if sub.empty:
            missing.append(z)
            continue
        row = sub.iloc[0].drop('Zip code')
        x = list(map(int, row.index))         # e.g. [1,2,…,12]
        y = row.values.astype(float)
        ax.plot(x, y, marker='o', linestyle='-', label=z)

    # if nothing got plotted
    if not ax.has_data():
        return render_template('graph.html',
                               data_sources=CSV_OPTIONS,
                               error="None of the entered ZIP codes were found.",
                               raw_input=raw,
                               current_src=selected_src)

    # Titles & labels
    ax.set_title(f"Comparison ({selected_src}): "
                 f"{', '.join([z for z in selected if z not in missing])}")
    ax.set_xlabel('Month')
    if selected_src == 'Rental':
        ax.set_ylabel('Monthly Cost in Dollars ($)')
    else:
        ax.set_ylabel('Price in Dollars ($)')
    ax.legend(title="ZIP code")

    # —— NEW: force x‑axis ticks Jan–Dec —— #
    month_nums   = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in month_nums]
    ax.set_xticks(month_nums)
    ax.set_xticklabels(month_labels)
    # ---------------------------------------- #

    plt.tight_layout()

    # render to PNG & base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)

    # re‑render template with the plot + any missing codes
    return render_template('graph.html',
                           data_sources=CSV_OPTIONS,
                           plot_data=img_b64,
                           selected=selected,
                           missing=missing,
                           raw_input=raw,
                           current_src=selected_src)

if __name__ == "__main__":
    app.run(debug=True)
