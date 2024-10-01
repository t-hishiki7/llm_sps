import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

def create_personality_visualizations(agents, base_folder):
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    y_attributes = ['strategy', 'movement', 'score']
    y_labels = ['Avg. Coop. Rate', 'Avg. Movement Distance', 'Final Score']

   
    plots_folder = os.path.join(base_folder, 'individual_plots')
    os.makedirs(plots_folder, exist_ok=True)

    fig = make_subplots(rows=3, cols=5, 
                        subplot_titles=[f'{trait}' for trait in traits],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.05)

    for i, (y_attribute, y_label) in enumerate(zip(y_attributes, y_labels)):
        for j, trait in enumerate(traits):
            x = [getattr(a.personality, trait) for a in agents]
            if y_attribute == 'strategy':
                y = [np.mean(a.strategy_history) for a in agents]
            elif y_attribute == 'movement':
                y = [np.mean(a.movement_history) for a in agents]
            else:
                y = [getattr(a, y_attribute) for a in agents]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([min(x), max(x)])
            line_y = slope * line_x + intercept
            
            hover_text = [f"Agent ID: {a.id}<br>" +
                          f"{trait.capitalize()}: {getattr(a.personality, trait):.2f}<br>" +
                          f"{y_label}: {y_val:.2f}<br>" +
                          f"Openness: {a.personality.openness:.2f}<br>" +
                          f"Conscientiousness: {a.personality.conscientiousness:.2f}<br>" +
                          f"Extraversion: {a.personality.extraversion:.2f}<br>" +
                          f"Agreeableness: {a.personality.agreeableness:.2f}<br>" +
                          f"Neuroticism: {a.personality.neuroticism:.2f}"
                          for a, y_val in zip(agents, y)]
            
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers', name=f'{trait} ({y_label})',
                           marker=dict(color='blue'),
                           text=hover_text, hoverinfo='text'),
                row=i+1, col=j+1
            )
            
            fig.add_trace(
                go.Scatter(x=line_x, y=line_y, mode='lines', name=f'{trait} regression ({y_label})',
                           line=dict(color='red', dash='dash')),
                row=i+1, col=j+1
            )
            
            fig.update_xaxes(title_text=trait.capitalize(), row=i+1, col=j+1)
            fig.update_yaxes(title_text=y_label, row=i+1, col=j+1)
            
            fig.add_annotation(
                xref='x domain', yref='y domain',
                x=0.05, y=0.95, 
                text=f'r = {r_value:.2f}',
                showarrow=False,
                row=i+1, col=j+1
            )

            plt.figure()
            plt.scatter(x, y, color='blue')
            plt.plot(line_x, line_y, color='red', linestyle='dashed')
            plt.xlabel(trait.capitalize())
            plt.ylabel(y_label)
            plt.title(f'{trait.capitalize()} vs {y_label}')
            plt.annotate(f'r = {r_value:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, color='black')
            plt.savefig(os.path.join(plots_folder, f'{y_attribute}_{trait}.png'))
            plt.close()

    fig.update_layout(height=1200, 
                      width=2000, 
                      showlegend=False,
                      hoverlabel=dict(            
                          bgcolor="white",            
                          font_size=35,            
                          font_family="Arial"        
                          )
    )

    fig.write_html(os.path.join(base_folder, 'personality_visualizations.html'))