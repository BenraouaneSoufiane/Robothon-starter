from __future__ import annotations

from dataclasses import dataclass

from .plan import ConstructionPlan, Task, Worker


ROBOT_FRIENDLY_SKILLS = {"carry_materials", "deliver_tools", "site_scan", "cleanup"}
HUMAN_ONLY_SKILLS = {"inspection", "masonry", "rebar", "electrical", "plumbing"}


@dataclass(frozen=True)
class AssignedTask:
    task: Task
    assignee: str
    kind: str
    reason: str


@dataclass(frozen=True)
class WorkSplit:
    owner_goal: str
    assignments: tuple[AssignedTask, ...]

    @property
    def robot_tasks(self) -> tuple[AssignedTask, ...]:
        return tuple(item for item in self.assignments if item.kind == "robot")

    @property
    def human_tasks(self) -> tuple[AssignedTask, ...]:
        return tuple(item for item in self.assignments if item.kind == "human")


def split_work(plan: ConstructionPlan, owner_goal: str) -> WorkSplit:
    assignments: list[AssignedTask] = []
    human_load: dict[str, int] = {worker.name: 0 for worker in plan.workers}

    for task in _topological_tasks(plan.tasks):
        if _robot_should_take(plan, task, owner_goal):
            assignments.append(
                AssignedTask(
                    task=task,
                    assignee=plan.robot.name,
                    kind="robot",
                    reason="matches robot capability and reduces repetitive or heavy human work",
                )
            )
            continue

        worker = _best_worker(plan.workers, human_load, task)
        human_load[worker.name] += task.effort
        assignments.append(
            AssignedTask(
                task=task,
                assignee=worker.name,
                kind="human",
                reason="needs human judgement, craft skill, or higher safety supervision",
            )
        )

    return WorkSplit(owner_goal=owner_goal, assignments=tuple(assignments))


def _robot_should_take(plan: ConstructionPlan, task: Task, owner_goal: str) -> bool:
    goal = owner_goal.lower()
    robot_can_do_it = task.required_skill in plan.robot.capabilities
    owner_wants_help = any(word in goal for word in ("heavy", "deliver", "carry", "cleanup", "scan", "robot"))
    safe_for_robot = task.risk != "high" and task.required_skill not in HUMAN_ONLY_SKILLS
    good_robot_work = task.required_skill in ROBOT_FRIENDLY_SKILLS or task.effort >= 3
    return robot_can_do_it and safe_for_robot and (owner_wants_help or good_robot_work)


def _best_worker(workers: tuple[Worker, ...], human_load: dict[str, int], task: Task) -> Worker:
    capable = [worker for worker in workers if task.required_skill in worker.skills]
    candidates = capable or list(workers)
    return min(candidates, key=lambda worker: (human_load[worker.name] / max(worker.max_effort, 1), human_load[worker.name]))


def _topological_tasks(tasks: tuple[Task, ...]) -> tuple[Task, ...]:
    remaining = {task.id: task for task in tasks}
    ordered: list[Task] = []
    completed: set[str] = set()

    while remaining:
        ready = [task for task in remaining.values() if set(task.dependencies) <= completed]
        if not ready:
            cycle = ", ".join(sorted(remaining))
            raise ValueError(f"task dependency cycle detected: {cycle}")
        ready.sort(key=lambda task: (len(task.dependencies), task.id))
        for task in ready:
            ordered.append(task)
            completed.add(task.id)
            del remaining[task.id]

    return tuple(ordered)
