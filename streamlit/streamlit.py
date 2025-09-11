import streamlit as st
from statsbombpy import sb
import mplsoccer as mpl
from kloppy import metrica
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import ListedColormap

def bulid_cmap(x, y):
    r,g,b = x
    r_, g_, b_ = y
    N = 256
    A = np.ones((N, 4))
    A[:, 0] = np.linspace(r, 1, N)
    A[:, 1] = np.linspace(g, 1, N)
    A[:, 2] = np.linspace(b, 1, N)
    cmp = ListedColormap(A)
    
    B = np.ones((N, 4))
    B[:, 0] = np.linspace(r_, 1, N)
    B[:, 1] = np.linspace(g_, 1, N)
    B[:, 2] = np.linspace(b_, 1, N)
    cmp_ = ListedColormap(B)
    
    newcolors = np.vstack((cmp(np.linspace(0, 1, 128)),
                            cmp_(np.linspace(1, 0, 128))))
    return ListedColormap(newcolors)
GRID_X, GRID_Y = 16, 12
pitch_x, pitch_y = 120, 80  # Standard pitch dimensions in meters
x_bins = np.linspace(0, pitch_x, GRID_X + 1)
y_bins = np.linspace(0, pitch_y, GRID_Y + 1)
xT = np.zeros((GRID_X, GRID_Y))
T = np.zeros((GRID_X, GRID_Y, GRID_X, GRID_Y))
S = np.zeros((GRID_X, GRID_Y))  # shot counts
M = np.zeros((GRID_X, GRID_Y))  # move counts
G = np.zeros((GRID_X, GRID_Y)) # goal probability
blue, red = (44,123,182), (215,25,28)
blue = [x/256 for x in blue]
red = [x/256 for x in red]
diverging = bulid_cmap(blue, red)
diverging_r = bulid_cmap(red, blue)

figsize = (9, 6)

def get_grid_cell(x, y):
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    return x_idx, y_idx

def get_events():
    comps = sb.competitions()
    comps = comps[comps['country_name'] == 'Spain']
    matches = sb.matches(competition_id=11, season_id=90).sort_values('match_date', ascending=False)
    matches = matches[(matches['home_team'] == 'Barcelona')]

    all_match_ids = matches.match_id.to_list()[:15]

    all_events = pd.DataFrame()

    for match_id in all_match_ids:
        match_events = sb.events(match_id=match_id)
        all_events = pd.concat([all_events, match_events], ignore_index=True)

    return all_events

def get_actions(all_events):
    passes = all_events[all_events['type'] == 'Pass']
    carries = all_events[all_events['type'] == 'Carry']
    shots = all_events[all_events['type'] == 'Shot']

    return passes, carries, shots

def process_movements():
    # Process passes and carries
    for _, row in pd.concat([passes, carries]).iterrows():
        try:
            start_x, start_y = row['location']
            end = row.get('pass_end_location') if isinstance(row.get('pass_end_location'), list) else row.get('carry_end_location')
            if not isinstance(end, list):
                continue
            end_x, end_y = end
            sx, sy = get_grid_cell(start_x, start_y)
            ex, ey = get_grid_cell(end_x, end_y)
            if 0 <= sx < GRID_X and 0 <= sy < GRID_Y and 0 <= ex < GRID_X and 0 <= ey < GRID_Y:
                T[sx, sy, ex, ey] += 1
                M[sx, sy] += 1
        except:
            continue

def process_shots():
    # Process shots
    for _, row in shots.iterrows():
        try:
            x, y = row['location']
            sx, sy = get_grid_cell(x, y)
            S[sx, sy] += 1
            if row['shot_outcome'] == 'Goal':
                G[sx, sy] += 1
        except:
            continue

def compute_xT(n_iterations=10):

    xT = np.zeros((GRID_X, GRID_Y))
    iterations = [xT]

    for iteration in range(n_iterations):
        new_xT = np.copy(xT)
        total_payoff = np.zeros((GRID_X, GRID_Y))
        gs = P_goal * P_shot
        for y in range(GRID_X):
            for x in range(GRID_Y):
                for q in range(GRID_X):
                    for z in range(GRID_Y):
                        total_payoff[y, x] += P_trans[y,x,q,z] * xT[q,z]
        new_xT = gs + (P_move * total_payoff)

        xT = new_xT

        iterations.append(xT) 
    
    return iterations
    


all_events = get_events()
passes, carries, shots = get_actions(all_events)
process_movements()
process_shots()

# Normalize transition probabilities
P_move = np.divide(M, M + S, out=np.zeros_like(M), where=(M + S) != 0)
P_shot = np.divide(S, M + S, out=np.zeros_like(S), where=(M + S) != 0)
P_goal = np.divide(G, S, out=np.zeros_like(G), where=S != 0)
P_trans = np.divide(T, T.sum(axis=(2,3), keepdims=True), out=np.zeros_like(T), where=T.sum(axis=(2,3), keepdims=True)!=0)


iterations = compute_xT()



st.title("Expected Threat (xT) for Barcelona 2020-2021")

time_range = st.slider("Select iterations", min_value=0, max_value=10, value=0)
current_iteration = iterations[time_range]

pitch = mpl.Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
fig, ax = pitch.draw(figsize=(10, 7))

bin_statistic = pitch.bin_statistic(
    np.repeat(x_bins[:-1], GRID_Y) + (pitch_x / GRID_X / 2),
    np.tile(y_bins[:-1], GRID_X) + (pitch_y / GRID_Y / 2),
    values=current_iteration.flatten(),
    statistic='mean',
    bins=(GRID_X, GRID_Y)
)

pitch.heatmap(bin_statistic, ax=ax, cmap='Greens', edgecolors='grey', alpha=0.75)
ax.set_title(f"xT - Iteration {time_range}", fontsize=16)
st.pyplot(fig)