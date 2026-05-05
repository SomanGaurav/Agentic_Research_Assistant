# executor.py
from crewai import Crew, Process
from task_registry import TASK_REGISTRY
from agents.manager_agent import make_manager_agent


def resolve_inputs(inputs: dict, context: dict) -> dict:
    """
    Replace $task_N_output references with actual outputs from prior tasks.
    Example: {"query": "$task_1_output"} → {"query": "<actual search result>"}
    """
    resolved = {}
    for key, value in inputs.items():
        if isinstance(value, str) and value.startswith("$"):
            ref_key = value[1:]  # strip leading $
            resolved[key] = context.get(ref_key, value)
        else:
            resolved[key] = value
    return resolved


def execute_plan(plan: dict, initial_context: dict = {}) -> str:
    context = {**initial_context}
    built_tasks = []
    task_index = {}     # task_id → Task object, used for context wiring

    print(f"\nIntent: {plan['intent_summary']}")
    print(f" Tasks planned: {len(plan['tasks'])}\n")

    for step in plan["tasks"]:
        task_id = step["task_id"]
        name = step["name"]

        if name not in TASK_REGISTRY:
            print(f"  ⚠ Unknown task '{name}', skipping.")
            continue

        resolved_inputs = resolve_inputs(step.get("inputs", {}), context)

        # Build the CrewAI Task from the registry factory
        task = TASK_REGISTRY[name](**resolved_inputs)

        # Wire all prior tasks as context so this task sees their outputs
        prior_tasks = list(task_index.values())
        if prior_tasks:
            task.context = prior_tasks

        built_tasks.append(task)
        task_index[task_id] = task

        print(f"  ✓ Queued [{task_id}] {name} — {step['reason']}")

    # Collect unique agents from the built tasks only
    # (no unused agents are loaded into the crew)
    agents = list({task.agent for task in built_tasks})

    # Build manager with awareness of exactly which agents are in this crew



    crew = Crew(
        agents=agents,
        tasks=built_tasks,
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()