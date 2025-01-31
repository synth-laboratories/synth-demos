import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tomllib
from craftaxlm import CraftaxACI, CraftaxClassicACI
from craftaxlm.recording import EpisodeRecorder
from synth_sdk.tracing.abstractions import RewardSignal
from tqdm.asyncio import tqdm as tqdm_asyncio
from zyk import LM

from crafter_agent.simple_react_agent import SimpleReActLanguageAgent

# Create videos directory at startup
full_video_dir = Path("recordings/crafter_agent")
full_video_dir.mkdir(parents=True, exist_ok=True)
video_dir = Path("crafter_agent")


async def run_agent_episodes(
    model_name: str,
    mode: str,
    seeds: List[int],
    max_steps: int,
    save_upload: bool = False,
) -> Tuple[Dict[int, Dict], Dict[int, List], Dict[int, bytes], Optional[float]]:
    os.environ["SYNTH_LOGGING_MODE"] = "instant"
    assert (
        os.getenv("SYNTH_API_KEY") is not None
    ), "SYNTH_API_KEY environment variable must be set"
    assert (
        os.getenv("OPENAI_API_KEY") is not None
    ), "OPENAI_API_KEY environment variable must be set"

    print(
        f"Running agent episodes fully in parallel (model: {model_name}, mode: {mode}, num_episodes: {len(seeds)})"
    )
    # Initialize agents and envs
    agents = {}
    envs = {}
    recorders = {}
    steps_data = {}
    done_flags = {}
    all_achievements = {}
    all_videos = {}
    reward_signals = []
    total_rewards = 0

    for seed in seeds:
        if mode == "classic":
            envs[seed] = CraftaxClassicACI(seed=seed, verbose=False)
        else:
            envs[seed] = CraftaxACI(seed=seed, verbose=False)

        lm = LM(
            model_name=model_name,
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            synth_logging=True,
        )
        agents[seed] = SimpleReActLanguageAgent(
            lm=lm, mode=mode, config={"max_history": 5, "max_agent_steps": max_steps}
        )
        recorders[seed] = EpisodeRecorder(enabled=True)
        steps_data[seed] = []
        done_flags[seed] = False

        # Record initial observation
        initial_obs = {"state": envs[seed].starting_obs}
        await agents[seed].add_observations([initial_obs])
        steps_data[seed].append(
            {
                "observation": initial_obs,
                "action": None,
                "reward": 0.0,
                "done": False,
                "achievements": [],
            }
        )
        recorders[seed].record_frame(envs[seed].state)

    progress_bars = {}
    for i, seed in enumerate(seeds):
        progress_bars[seed] = tqdm_asyncio(
            range(max_steps),
            desc=f"\033[92m⚡ Ep {seed:02d}\033[0m",
            position=i,
            leave=True,
            total=max_steps,
            ascii=False,
            ncols=100,
            colour="green",
            dynamic_ncols=True,
            bar_format="{desc:<12} Step {n_fmt:>3}/{total_fmt} "
            "({percentage:3.0f}%) \033[1m\033[38;5;39m{bar:30}\033[0m "
            "{elapsed}<{remaining}",
        )

    async def run_episode(seed: int, progress_bar):
        env = envs[seed]
        agent = agents[seed]
        rec = recorders[seed]
        steps_completed = 0

        while steps_completed < max_steps and not done_flags[seed]:
            try:
                # Create a task for getting actions but don't await it immediately
                action_task = asyncio.create_task(agent.get_actions())

                # Allow other coroutines to run while waiting
                actions_list = await action_task

                if not actions_list:
                    # Give other coroutines a chance to run
                    await asyncio.sleep(0)
                    continue

                for action in actions_list:
                    if done_flags[seed]:
                        break

                    try:
                        mapped = env.map_action_string_to_int(action)
                        step_info = env._step(mapped)
                        steps_data[seed].append(
                            {
                                "observation": step_info,
                                "action": action,
                                "reward": step_info["reward"],
                                "done": step_info["done"],
                                "achievements": env.achievement_deltas[-1],
                            }
                        )
                        rec.record_frame(env.state)

                        # Create a task for adding observations but don't block
                        obs_task = asyncio.create_task(
                            agent.add_observations([step_info])
                        )
                        await obs_task

                        if step_info["done"]:
                            done_flags[seed] = True
                            break
                    except:
                        pass

                steps_completed += 1
                progress_bar.update(1)

                # Give other coroutines a chance to run
                await asyncio.sleep(0)

            except Exception as e:
                done_flags[seed] = True
                break

        # Handle episode completion
        raw_achievements = env.terminate()
        final_achievements = {k: bool(v) for k, v in raw_achievements.items()}
        total_achievements = sum(map(int, final_achievements.values()))
        random_suffix = uuid.uuid4()

        reward_signal = RewardSignal(
            question_id=f"episode_{seed}_{random_suffix}",
            system_instance_id=agent.system_instance_id,
            reward=total_achievements,
            annotation=str(final_achievements),
        )
        reward_signals.append(reward_signal)

        # Save video
        video_path = video_dir / f"episode_{seed}.mp4"
        rec.record_frame(env.state)
        rec.save_video(video_path, fps=3)
        with open(full_video_dir / f"episode_{seed}.mp4", "rb") as f:
            video_bytes = f.read()

        return {
            "achievements": final_achievements,
            "reward": total_achievements,
            "video": video_bytes,
        }

    # Create and run tasks for all episodes
    episode_tasks = []
    for seed in seeds:
        task = asyncio.create_task(run_episode(seed, progress_bars[seed]))
        episode_tasks.append(task)

    # Wait for all episodes to complete and gather results
    episode_results = await asyncio.gather(*episode_tasks)

    # Process results
    total_rewards = 0
    for seed, result in enumerate(episode_results):
        all_achievements[seed] = result["achievements"]
        all_videos[seed] = result["video"]
        total_rewards += result["reward"]

    # Close all progress bars
    for seed in seeds:
        progress_bars[seed].close()

    # Print average score
    average_score = None
    if not save_upload and seeds:
        average_score = total_rewards / len(seeds)
        print(f"\nAverage score across {len(seeds)} episodes: {average_score:.2f}")

    return all_achievements, steps_data, all_videos, average_score


def load_config(config_path: str = "crafter_agent/config.toml") -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    import asyncio

    config = load_config()
    asyncio.run(
        run_agent_episodes(
            model_name=config["agent"]["model_name"],
            mode=config["agent"]["mode"],
            seeds=config["environment"]["seeds"],
            max_steps=config["agent"]["max_steps"],
            save_upload=config["agent"]["save_upload"],
        )
    )
