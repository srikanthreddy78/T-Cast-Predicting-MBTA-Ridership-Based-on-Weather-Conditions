import pandas as pd
import plotly.graph_objects as go
import numpy as np

#station coords
station_coords = {
'Alewife': (13,86),
'Davis': (14.3,83),
'Porter': (15.600000000000001,80),
'Harvard': (16.900000000000002,77),
'Central': (18.200000000000003,74),
'Kendall/MIT': (19.500000000000004,71),
'Charles/MGH': (20.800000000000004,67),
'Park Street': (22,63),
'Downtown Crossing': (23,60),

'Wonderland': (34,91),
'Revere Beach': (33,88),
'Beachmont': (32,85),
'Suffolk Downs': (31,82),
'Orient Heights': (30,79),
'Wood Island': (29,76),
'Airport': (28,73),
'Maverick': (27,70),
'Aquarium': (26,67),
'State Street': (25,64),

'Oak Grove': (24,92.5),
'Malden Center': (24,90.0),
'Wellington': (24,87.5),
'Assembly': (24,85.0),
'Sullivan Square': (24,82.5),
'Community College': (24,80.0),

'Chinatown': (22,56),
'Tufts Medical Center': (21,53),
'Back Bay': (20,50),
'Massachusetts Avenue': (19,47),
'Ruggles': (18,44),
'Roxbury Crossing': (17,41),
'Jackson Square': (16,38),
'Stony Brook': (15,35),
'Green Street': (14,32),
'Forest Hills': (13,29),

'Medford/Tufts': (18,90),
'Ball Square': (18.7,88),
'Magoun Square': (19.4,86),
'Gilman Square': (20.099999999999998,84),
'East Somerville': (20.799999999999997,82),

'Lechmere': (23, 77),
'Science Park': (23.7, 75),
'North Station': (24, 72),
'Haymarket': (24, 70),

'Boylston': (21,60),
'Arlington': (19,59),
'Copley': (17.5,59),
'Hynes Convention Center': (16,59),
'Kenmore': (14.5,59),

'Broadway': (26,47),
'Andrew': (26,44),
'JFK/UMass': (26,41),

'Bowdoin': (22, 69),
'Government Center': (23, 67), 
'South Station': (25, 56),
'Prudential': (16.5, 56),
'Symphony': (16.5, 52),

'North Quincy': (28,29),
'Wollaston': (29,26),
'Quincy Center': (30,23),
'Quincy Adams': (30.5,20),
'Braintree': (30.5, 17),

'Savin Hill': (24,29),
'Fields Corner': (24,27),
'Shawmut': (24,25),
'Ashmont': (24,23),

'Riverside': (6, 38),
'Mattapan Line': (20, 18),
'Union Square': (20, 79),

}

entries_raw = pd.read_csv('mbta_test_predictions.csv', parse_dates=['service_date'])
start, end = pd.Timestamp('2022-03-02'), pd.Timestamp('2023-03-02')
entries = entries_raw[
    (entries_raw['service_date'] >= start) &
    (entries_raw['service_date'] <= end) &
    (entries_raw['station_name'].isin(station_coords))
].assign(
    date=lambda d: d['service_date'].dt.strftime('%Y-%m-%d'),
    x=lambda d: d['station_name'].map(lambda s: station_coords[s][0]),
    y=lambda d: d['station_name'].map(lambda s: station_coords[s][1])
)

df = entries.rename(columns={'station_name': 'station'})[
    ['date','station','x','y','tavg','prcp','wspd','actual_entries','predicted_entries']
]
df['error'] = (df['predicted_entries'] - df['actual_entries']).abs()

max_actual = df['actual_entries'].max()
max_px = 40

stations = list(station_coords.keys())
xs = [station_coords[s][0] for s in stations]
ys = [station_coords[s][1] for s in stations]

dates = sorted(df['date'].unique())
first = dates[0]
df0 = df[df['date'] == first].set_index('station')

sizes0 = [ (df0.at[s,'actual_entries'] if s in df0.index else 0) / max_actual * max_px for s in stations ]
errs0  = [ (df0.at[s,'error'] if s in df0.index else 0) for s in stations ]
acts0  = [ (df0.at[s,'actual_entries'] if s in df0.index else 0) for s in stations ]
preds0 = [ (df0.at[s,'predicted_entries'] if s in df0.index else 0) for s in stations ]

scatter0 = go.Scatter(
    x=xs, y=ys, mode='markers',
    marker=dict(size=sizes0, color=errs0, coloraxis='coloraxis'),
    text=stations,
    customdata=np.stack([acts0, errs0, preds0], axis=-1),
    hovertemplate="%{text}<br>Actual: %{customdata[0]:.0f}<br>Predicted:%{customdata[2]:.0f}<br>Error: %{customdata[1]:.0f}<extra></extra>"
)

layout = go.Layout(
    title=f"Entries on {first}",
    xaxis=dict(visible=False, range=[0,100]),
    yaxis=dict(visible=False, range=[0,100]),
    template='plotly_white',
    coloraxis=dict(
        colorscale=[[0,'green'],[1,'red']],
        cmin=0, cmax=5000,
        colorbar=dict(x=1.02, y=0.5, title='Error')
    )
)

frames = []
for date in dates:
    grp = df[df['date'] == date].set_index('station')
    sizes = [ (grp.at[s,'actual_entries'] if s in grp.index else 0) / max_actual * max_px for s in stations ]
    errs  = [ (grp.at[s,'error']          if s in grp.index else 0) for s in stations ]
    acts  = [ (grp.at[s,'actual_entries'] if s in grp.index else 0) for s in stations ]
    preds = [ (grp.at[s,'predicted_entries'] if s in grp.index else 0) for s in stations ]
    row   = grp.iloc[0] if not grp.empty else None
    weather = f"{row['tavg']:.1f}°F | {row['prcp']:.2f}\" rain | {row['wspd']:.1f}mph" if row is not None else ''
    frames.append(go.Frame(
        name=date,
        data=[dict(marker=dict(size=sizes, color=errs, coloraxis='coloraxis'), customdata=np.stack([acts, errs, preds], axis=-1))],
        layout=go.Layout(title_text=f"Entries on {date} — {weather}"),
        traces=[0]
    ))

fig = go.Figure(data=[scatter0], layout=layout, frames=frames)
fig.update_layout(
    updatemenus=[dict(type='buttons', showactive=False,
        x=1.1,y=1.05,xanchor='right',yanchor='top',
        buttons=[dict(label='▶ Play',method='animate',args=[None,{'frame':{'duration':200,'redraw':True},'fromcurrent':True}])]
    )],
    sliders=[dict(active=0,pad={'t':50},
        steps=[dict(method='animate',args=[[fr.name],{'frame':{'duration':0,'redraw':True},'mode':'immediate'}],label=fr.name) for fr in frames]
    )]
)

fig.show()
