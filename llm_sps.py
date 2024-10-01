import numpy as np
import random
from ollama import Client
import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from sps_visualization_functions import create_personality_visualizations
from pyline_notify import notify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

USE_PERSONALITY = True
USE_NEIGHBOR_PERSONALITY = False  

MODEL = "gemma2"
LINE_TOKEN = os.getenv("LINE_TOKEN")

# Constants and Global Variables
N = 50  # Number of agents
T = 1  # Total simulation steps,
W = 100  # World size
SEED = 101
R = 20  # Interaction radius
SPEED = 7
PMUT = 0.03  # Probability of random strategy flip

# Payoff matrix
PR = 1.2  # Reward for mutual cooperation
PT = 1.4  # Temptation to defect
PS = -1.4  # Sucker's payoff
PP = -1.0  # Punishment for mutual defection
PF = [PP, PS, PT, PR]

colorlist = ['red', 'blue']


def clip(x):
    return x % W

def payoff(s1, s2):
    return PF[int(s2*2 + s1)]

# Classes
class Personality:
    def __init__(self):
        self.openness = round(random.randint(0, 10) * 0.1, 1)
        self.conscientiousness = round(random.randint(0, 10) * 0.1, 1)
        self.extraversion = round(random.randint(0, 10) * 0.1, 1)
        self.agreeableness = round(random.randint(0, 10) * 0.1, 1)
        self.neuroticism = round(random.randint(0, 10) * 0.1, 1)

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.x = random.randint(0, W-1)
        self.y = random.randint(0, W-1)
        self.state = random.randint(0, 1)
        self.personality = Personality() if USE_PERSONALITY else None
        self.x_next = self.x
        self.y_next = self.y
        self.score = 0  
        self.action = [0, 0]
        self.reasoning = ""
        self.payoff = 0
        self.strategy_history = []
        self.movement_history = []

    def get_neighbors_info(self):
        neighbors_info = []
        for a in agents:
            if a != self:
                dx = (a.x - self.x + W // 2) % W - W //2
                dy = (a.y - self.y + W // 2) % W - W //2 
                distance = np.sqrt(dx**2 + dy**2)
                if distance <= R:
                    angle = int(np.degrees(np.arctan2(dy, dx)) + 360) % 360
                    neighbor_info = {
                        "Distance to the neighbor": round(distance, 2),
                        "Directions to the neighbor": angle,
                        "State of the neighbor": "Cooperate" if a.state == 1 else "Defect",
                    }
                    if USE_NEIGHBOR_PERSONALITY:
                        neighbor_info["Personality of the neighbor"] = vars(a.personality)
                    neighbors_info.append(neighbor_info)
        return neighbors_info
    
    def calc(self):
        neighbors_info = self.get_neighbors_info()
        context = {
            "current_strategy": "Cooperate" if self.state == 1 else "Defect",
            "neighbors": neighbors_info,
            "interaction_radius": R
        }

        prompt = f""" 
        Your goal is to maximize your total payoff by strategically choosing to Cooperate or Defect with other participants, and by moving effectively within the 2D environment.
        You will make decisions each round, but you should consider the impact of your actions on future rounds.

        #Explanation of the SPS model
        **1. Environment:**
        * You are represented as a particle in a 2D space.
        * You can interact with particles within your interaction radius (R = {R}).

        **2. Interaction rules:**
        - Payoff structure: Based on the Prisoner's Dilemma game.
         * Both Cooperate (CC): Both gain moderate positive payoff ({PR})
         * Both Defect (DD): Both receive small negative payoff ({PP})
         * One Cooperates, One Defects (CD):
           - Cooperator: Severe negative payoff ({PS}) (sucker's payoff)
           - Defector: Large positive payoff ({PT}) (temptation payoff)
        - Important: The payoff gained in one step is divided by the distance between particles. This means interactions with closer particles have a greater impact.

        **3. Your Task:**
        * In each round, you will make two simultaneous decisions:
        *   a. Strategy: Choose to Cooperate or Defect with nearby particles.
              - You may choose to defect against cooperators and move closer to them to maximize your own payoff.
              - Alternatively, you may choose to cooperate with other cooperators and move closer to them to improve both your payoffs.
        
        *   b. Movement: Decide how far to move from your current position in the 2D space.
              - You can move towards particles that have brought you positive payoff.
              - You can move away from particles that have caused you negative payoff.
        * Each round, choose your *Strategy* and *Movement*:

        **4. Objective:** Maximize your total payoff at the end of the game. Consider not only the current round's payoff but also the potential payoffs in future rounds.

        **5. Current State and Context:**
        {json.dumps(context, indent=2)}

        {"**6. Personality Traits (0.0 - 1.0):**\n" + json.dumps(vars(self.personality), indent=2) if USE_PERSONALITY else ""}

        **Task:** Based on the rules, your current context{" and personality traits" if USE_PERSONALITY else ""}, determine your next action.  Consider the positions of neighboring particles and devise a long-term strategy.

        **Response Format (Strictly adhere to this format):**
        Action: [magnitude, direction] (magnitude should be between 0 and {SPEED}, direction should be between 0 and 360 degrees)
        Strategy: (Cooperate/Defect)
        Reasoning: Provide a concise explanation (1-2 sentences) for your decision, focusing on:
           {"- How it aligns with your personality traits\n" if USE_PERSONALITY else ""}           
            - How it responds to the current context"
        """

        try:
            response = client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a participant in a Social Particle Swarm (SPS) experiment."},
                    {"role": "user", "content": prompt}
                ],
            )
            response_text = response['message']['content'] 
            action, strategy, reasoning = self.parse_llm_response(response_text)
            
            self.action = action
            self.x_next, self.y_next = self.polar_to_cartesian(action[0], action[1])
            self.state_next = 1 if strategy.lower() == "cooperate" else 0
            self.reasoning = reasoning

        except Exception as e:
            print(f"Error in LLM processing for Agent {self.id}: {e}")
            self.x_next = 0
            self.y_next = 0
            self.action = [0, 0]
            self.state_next = self.state  
            self.reasoning = "No movement due to LLM processing error. Current state maintained."

    def parse_llm_response(self, response_text):
        lines = response_text.split('\n')
        action = [0, 0]
        strategy = ""
        reasoning = ""

        for line in lines:
            if line.startswith("Action:"):
                action_str = line.split(":")[1].strip()
                action = [float(x) for x in action_str.strip('[]').split(',')]
                action[0] = max(0, min(action[0], SPEED))
                action[1] = action[1] % 360
            elif line.startswith("Strategy:"):
                strategy = line.split(":")[1].strip()
            elif line.startswith("Reasoning:"):
                reasoning = ':'.join(line.split(':')[1:]).strip()

        return action, strategy, reasoning

    def polar_to_cartesian(self, magnitude, direction):
        angle_rad = np.radians(direction)
        x = magnitude * np.cos(angle_rad)
        y = magnitude * np.sin(angle_rad)
        return x, y

    def move(self):
        self.x += self.x_next
        self.y += self.y_next
        
        self.x = clip(self.x)
        self.y = clip(self.y)
        
        self.state = self.state_next if random.random() < 1.0 - PMUT else 1 - self.state_next
        
        self.strategy_history.append(self.state)
        self.movement_history.append(self.action[0])

    def calculate_payoff(self):
        self.payoff = 0
        for a in agents:
            if a != self:
                dx = (a.x - self.x + W // 2) % W - W //2
                dy = (a.y - self.y + W // 2) % W - W //2
                distance = np.sqrt(dx**2 + dy**2)
                if distance <= R:
                    if distance == 0:
                        distance = 0.1
                    self.payoff += payoff(self.state, a.state) / distance

        self.score += self.payoff 

    def get_personality_str(self):
        if self.personality:
            return f"O:{self.personality.openness:.1f}, C:{self.personality.conscientiousness:.1f}, E:{self.personality.extraversion:.1f}, A:{self.personality.agreeableness:.1f}, N:{self.personality.neuroticism:.1f}"
        return "N/A"

def create_animation(df):
    time_steps = df['time'].unique()
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Agent Positions', 'Cooperation Ratio'),
                        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])
    
    def format_reasoning(reasoning, max_line_length=80):
        words = reasoning.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > max_line_length:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        if current_line:
            lines.append(' '.join(current_line))
        return '<br>'.join(lines)

    def create_hover_text(row):
        formatted_reasoning = format_reasoning(row['reasoning'])
        return (f"<b>ID:</b> {row['agent_id']}<br>"
                f"<b>State:</b> {'Cooperate' if row['state'] == 1 else 'Defect'}<br>"
                f"<b>Big Five:</b> {row['personality']}<br>"
                f"<b>Action:</b> Distance={row['action_magnitude']:.2f}, Angle={row['action_direction']:.2f}°<br>"
                f"<b>Score:</b> {row['score']:.2f}<br>"
                f"<b>Payoff:</b> {row['payoff']:.2f}<br>"
                f"<b>Reasoning:</b><br>{formatted_reasoning}")
    
    min_score = df['score'].min()
    max_score = df['score'].max()
    def score_to_size(score):
        return 10 + (score - min_score) / (max_score - min_score) * 15

    scatter = go.Scatter(
        x=df[df['time'] == 0]['x'],
        y=df[df['time'] == 0]['y'],
        mode='markers',
        marker=dict(
            size=df[df['time'] == 0]['score'].apply(score_to_size),
            #size=12,
            color=df[df['time'] == 0]['state'],
            colorscale=['red', 'blue'],
            showscale=False
        ),
        text=df[df['time'] == 0].apply(create_hover_text, axis=1),
        hoverinfo='text'
    )
    fig.add_trace(scatter, row=1, col=1)
    
    cooperation_ratio = df.groupby('time')['state'].mean()
    line = go.Scatter(
        x=cooperation_ratio.index,
        y=cooperation_ratio.values,
        mode='lines+markers'
    )
    fig.add_trace(line, row=1, col=2)
    
    fig.update_layout(
        title='Agent Simulation',
        xaxis=dict(range=[0, W], title='X'),
        yaxis=dict(range=[0, W], title='Y'),
        xaxis2=dict(title='Time Step'),
        yaxis2=dict(title='Cooperation Ratio', range=[0, 1]),
        hoverlabel=dict(
            bgcolor="white",
            font_size=35,
            font_family="Arial"
        )
    )
    
    frames = [go.Frame(
        data=[
            go.Scatter(
                x=df[df['time'] == t]['x'],
                y=df[df['time'] == t]['y'],
                mode='markers',
                marker=dict(
                    size=df[df['time'] == t]['score'].apply(score_to_size),  
                    #size=12,
                    color=df[df['time'] == t]['state'],
                    colorscale=['red', 'blue'],
                    showscale=False
                ),
                text=df[df['time'] == t].apply(create_hover_text, axis=1),
                hoverinfo='text'
            ),
            go.Scatter(
                x=cooperation_ratio.index[:t+1],
                y=cooperation_ratio.values[:t+1],
                mode='lines+markers'
            )
        ],
        traces=[0, 1],
        name=f'frame{t}'
    ) for t in time_steps]
    
    fig.frames = frames
    
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                     dict(label='Pause',
                          method='animate',
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])]
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f'frame{k}'], dict(mode='immediate', frame=dict(duration=100, redraw=True))],
                label=f'{k}'
            ) for k in range(len(time_steps))],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix='Time Step: ', visible=True, xanchor='right'),
            len=0.9
        )]
    )
    
    return fig

@notify(LINE_TOKEN)
def main():
    global agents, client
    random.seed(SEED)
    agents = [Agent(i) for i in range(N)]
    client = Client(host='http://localhost:11434') 

    start_time = time.time()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join("result", f"{timestamp}")
    os.makedirs(base_folder, exist_ok=True)
    
    all_data = []
    for t in range(T):
        print(f"Step{t}")
        for agent in agents:
            agent.calc()
        
        for agent in agents:
            agent.move()
            agent.calculate_payoff()
        
        frame_data = pd.DataFrame({
            'time': [t] * N,
            'agent_id': [a.id for a in agents],
            'x': [a.x for a in agents],
            'y': [a.y for a in agents],
            'state': [a.state for a in agents],
            'score': [a.score for a in agents],
            'payoff': [a.payoff for a in agents],
            'action_magnitude': [a.action[0] for a in agents],
            'action_direction': [a.action[1] for a in agents],
            'reasoning': [a.reasoning for a in agents],
            'personality': [a.get_personality_str() for a in agents]
        })
        all_data.append(frame_data)

    df = pd.concat(all_data, ignore_index=True)
    csv_path = os.path.join(base_folder, 'agent_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Simulation completed. Data saved to '{csv_path}'.")

    fig = create_animation(df)
    html_path = os.path.join(base_folder, "sps_animation.html")
    fig.write_html(html_path)
    print(f"Animation saved to '{html_path}'.")

    end_time = time.time()
    execution_time = end_time - start_time
    save_experiment_parameters(base_folder, execution_time)
    if USE_PERSONALITY:
        create_personality_visualizations(agents, base_folder)



def save_experiment_parameters(base_folder, execution_time):
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    parameters = {
        "USE_PERSONALITY": USE_PERSONALITY,
        "N": N,
        "T": T,
        "W": W,
        "SEED": SEED,
        "R": R,
        "SPEED": SPEED,
        "PMUT": PMUT,
        "PR": PR,
        "PT": PT,
        "PS": PS,
        "PP": PP,
        "MODEL": MODEL,
        "execution_time_seconds": execution_time,
        "execution_time_formatted": formatted_time
    }
    
    param_file = os.path.join(base_folder, "experiment_parameters.json")
    with open(param_file, 'w') as f:
        json.dump(parameters, f, indent=4)

if __name__ == "__main__":
    main()