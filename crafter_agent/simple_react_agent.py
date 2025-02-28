import os
import uuid
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from crafter_agent.game_info import (
    craftax_classic_action_dict,
    craftax_classic_game_rules,
    craftax_full_action_dict,
    craftax_full_game_rules,
    craftax_game_tips,
    crafter_game_tips,
)
from synth_sdk import AsyncOpenAI as SynthAsyncOpenAI
from synth_sdk import trace_event_async

load_dotenv()




class ReAct(BaseModel):
    reasoning: str
    actions: List[str]


DEFAULT_INSTRUCTIONS = """
# Objective
Please choose 3-5 actions to take next in the game, and provide a thought justifying the long-term end you'd like to achieve with these actions.
These actions will be executed sequentially in the game environment, and the results will be shared with you in the next round. 

# Constraints
- If your previous actions worked toward a valuable goal, continue in that direction unless you have a good reason to change course.
- If there's no important thought to share, don't share anything. "No update" is a valid and the most common response.
- Render your actions by name. All rendered names must match the names in the "Actions Available" section.
- If you attempt an action that requires certain conditions to be met without meeting those conditions, the action will fail.
- Never select 'Do' multiple times in a row unless fighting a mob or enemy. In other cases, it's a wasted action. 
    When you select 'Do', you will interact with whatever is one step in the direction you are facing (indicated under the Object_you_are_facing heading if there is anything) and zero steps away in every other direction. If there is nothing one step (e.g. one step up, one step down, one step right, or one step left) in the direction you are facing, your 'Do' action will fail.
        - If something is e.g. one step up and one step right, attempting to interact with it will fail. You must be only one step away in the direction you are facing.
- To change the direction you are facing, move in the direction you want to face.
"""


class SimpleReActLanguageAgent:
    obs_history: List[Dict]
    react_history: List[Dict]
    client: SynthAsyncOpenAI
    config: Dict

    def __init__(
        self,
        mode: Literal["craftax_classic", "craftax_full"] = "craftax_classic",
        config: Optional[Dict] = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ):
        self.obs_history = []
        self.react_history = []
        self.client = SynthAsyncOpenAI()
        self.mode = mode
        self.config = config or {"max_history": 5, "max_agent_steps": 10}
        self.instructions = instructions

        self.system_instance_id = str(uuid.uuid4())
        self.system_name = (
            f"CRAFTAX-TEST-REACT-DEMO-{os.getenv('DEMO_NAME', 'YOUR-NAME-HERE')}"
        )
        if "YOUR-NAME-HERE" in self.system_name:
            raise ValueError(
                "Please set the DEMO_NAME environment variable in your .env file"
            )

    def render_history(self):
        def remove_empty_error_section(d):
            if "errors" in d and not len(d["errors"]["illegal_actions"]) > 0:
                d.pop("errors")
            return d

        max_history = self.config.get("max_history", 5)

        react_history = [
            f"<{i} reasoning step(s) in the past>{remove_empty_error_section(item)}</{i} reasoning step(s) in the past>"
            for i, item in enumerate(reversed(self.react_history[-max_history:]), 1)
        ]
        obs_history = [
            f"<{i} environment step(s) in the past>{remove_empty_error_section(item)}</{i} environment step(s) in the past>"
            for i, item in enumerate(reversed(self.obs_history[-max_history:]), 1)
        ]
        return "\n".join(react_history), "\n".join(obs_history)

    @trace_event_async(
        event_type="re-act",
    )
    async def get_actions(self):
        if self.mode == "classic":
            rules = craftax_classic_game_rules
            game_tips = crafter_game_tips
            actions = craftax_classic_action_dict
        elif self.mode == "full":
            rules = craftax_full_game_rules
            game_tips = craftax_game_tips
            actions = craftax_full_action_dict
        else:
            raise ValueError(f"Mode {self.mode} not recognized")

        system_message = f"""
# Premise
You're playing the game of Crafter.
Here is some information about this setting
<Crafter Information>
<Rules>
{rules}
</Rules>
<Tips>
{game_tips}
</Tips>
<Actions Available>
{[a for a in list(actions.keys()) if a.lower() not in ['noop']]}
</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment

{self.instructions}
"""
        react_history, obs_history = self.render_history()
        user_message = f"""
# Recent Actions / Thoughts
{react_history}
# Recent Observations
{obs_history}

Your next actions / thought: """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = await self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ReAct,
            temperature=0.1,
        )
        react_step = response.choices[0].message.parsed
        illegal_actions = [
            action
            for action in react_step.actions
            if action not in craftax_classic_action_dict.keys()
        ]
        legal_actions = [
            action
            for action in react_step.actions
            if action in craftax_classic_action_dict.keys()
        ]
        react_info = react_step.dict()
        react_info["errors"] = {
            "illegal_actions": illegal_actions,
        }

        self.react_history.append(react_info)

        return legal_actions

    async def add_observations(self, observations: List[Dict]):
        for observation in observations:
            self.obs_history.append(observation["state"])  # [0].default_readout()
