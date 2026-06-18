from __future__ import annotations

import argparse

from .assignment import split_work
from .bricklaying import DEFAULT_BRICK_COUNT, run_bricklaying_demo
from .plan import load_plan
from .sim import robot_waypoints, run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and simulate a construction co-working robot.")
    parser.add_argument("plan", help="YAML construction plan")
    parser.add_argument("--owner-goal", help="What the owner wants help with")
    parser.add_argument("--dry-run", action="store_true", help="Only print the work split")
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=1.0, help="Robot route speed multiplier")
    parser.add_argument("--video", help="Write an MP4 video of the simulation to this path")
    parser.add_argument(
        "--bricklaying",
        action="store_true",
        help="Run the FF Master humanoid bricklaying demo",
    )
    parser.add_argument(
        "--bricks",
        type=int,
        default=DEFAULT_BRICK_COUNT,
        help="Number of bricks to lay in the bricklaying demo",
    )
    args = parser.parse_args()

    plan = load_plan(args.plan)
    owner_goal = args.owner_goal or input(f"{plan.owner}, what do you want the robot to help with? ")
    split = split_work(plan, owner_goal)

    print(f"\nProject: {plan.name}")
    print(f"Owner goal: {split.owner_goal}")
    print("\nHuman work:")
    for assignment in split.human_tasks:
        print(f"- {assignment.assignee}: {assignment.task.title} [{assignment.reason}]")

    print("\nRobot work:")
    for assignment in split.robot_tasks:
        zone = plan.zones[assignment.task.zone]
        print(f"- {assignment.assignee}: {assignment.task.title} at {zone.label} [{assignment.reason}]")

    waypoints = robot_waypoints(plan, split)
    if waypoints:
        print("\nRobot route:")
        for waypoint in waypoints:
            print(f"- {waypoint.label}: {waypoint.position}")

    if args.bricklaying and not args.dry_run:
        run_bricklaying_demo(render=args.render, speed=args.speed, video=args.video, brick_count=args.bricks)
    elif not args.dry_run:
        run_simulation(plan, split, render=args.render, speed=args.speed, video=args.video)


if __name__ == "__main__":
    main()
