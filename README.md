# synth-demos
Examples of how you can use Synth to improve agents


# Craftax Example - Error Analysis - Cuvier 
- Create an account at [usesynth.ai](https://www.usesynth.ai/signin)
- Go to the account page and create an API key. Add it to SYNTH_API_KEY in .env
- Save your OpenAI api key in .env
- Rename your agent to "CRAFTAX-DEMO-{Your Name}. Do that by checking simple_react_agent.py lines 64, 86
- run ```uv run python crafter_agent/run_agent.py```
- Check the 'agents' page in the Synth dashboard. You should see agent traces once your agent gets started!
- Once your agents are done, click "Trace Enrichment" and "Error Clustering" under the 'Analysis Schedules' section. This will direct Synth to evaluate your agent runs.
- In 30min - 1hr, review the 'Errors' page for results. Use the chat interface or Cuvier agent to investigate!