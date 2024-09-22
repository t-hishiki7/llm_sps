import pandas as pd
import os
from sps_visualization_functions import create_personality_visualizations

class Agent:
    def __init__(self, id, personality, strategy_history, movement_history, score):
        self.id = id
        self.personality = personality
        self.strategy_history = strategy_history
        self.movement_history = movement_history
        self.score = score

class Personality:
    def __init__(self, openness, conscientiousness, extraversion, agreeableness, neuroticism):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism

def load_agents_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    agents = []
    
    for agent_id in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent_id]
        
        personality_str = agent_data['personality'].iloc[0]
        personality_dict = {trait.split(':')[0]: float(trait.split(':')[1]) for trait in personality_str.split(', ')}
        
        personality = Personality(
            openness=personality_dict['O'],
            conscientiousness=personality_dict['C'],
            extraversion=personality_dict['E'],
            agreeableness=personality_dict['A'],
            neuroticism=personality_dict['N']
        )
        
        strategy_history = agent_data['state'].tolist()
        movement_history = agent_data['action_magnitude'].tolist()
        score = agent_data['score'].iloc[-1]
        
        agent = Agent(agent_id, personality, strategy_history, movement_history, score)
        agents.append(agent)
    
    return agents

def main():
    #任意のパス指定
    simulation_folder = "sps_simulation_20240919_022208"  

    csv_path = os.path.join(simulation_folder, "agent_data.csv")

    agents = load_agents_from_csv(csv_path)

    print("preparation OK")
    
    create_personality_visualizations(agents, simulation_folder)
    
    print(f"Personality visualizations were generated in {simulation_folder}.")

if __name__ == "__main__":
    main()