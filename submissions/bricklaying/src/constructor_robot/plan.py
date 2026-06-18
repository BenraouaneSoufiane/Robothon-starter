from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Zone:
    id: str
    label: str
    position: tuple[float, float]


@dataclass(frozen=True)
class Worker:
    name: str
    skills: frozenset[str]
    max_effort: int = 6


@dataclass(frozen=True)
class Robot:
    name: str
    capabilities: frozenset[str]
    payload_kg: int = 20


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    zone: str
    required_skill: str
    effort: int
    risk: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConstructionPlan:
    name: str
    owner: str
    site: str
    workers: tuple[Worker, ...]
    robot: Robot
    zones: dict[str, Zone]
    tasks: tuple[Task, ...]


def load_plan(path: str | Path) -> ConstructionPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("plan file must contain a YAML mapping")
    return parse_plan(raw)


def parse_plan(raw: dict[str, Any]) -> ConstructionPlan:
    project = _mapping(raw, "project")
    workers = tuple(_parse_worker(item) for item in _sequence(raw, "workers"))
    robot = _parse_robot(_mapping(raw, "robot"))
    zones = {_parse_zone(item).id: _parse_zone(item) for item in _sequence(raw, "zones")}
    tasks = tuple(_parse_task(item) for item in _sequence(raw, "tasks"))

    _validate_references(zones, tasks)

    return ConstructionPlan(
        name=str(project.get("name", "Untitled project")),
        owner=str(project.get("owner", "Owner")),
        site=str(project.get("site", "Unknown site")),
        workers=workers,
        robot=robot,
        zones=zones,
        tasks=tasks,
    )


def _parse_worker(raw: Any) -> Worker:
    data = _ensure_mapping(raw, "worker")
    return Worker(
        name=str(data["name"]),
        skills=frozenset(str(skill) for skill in data.get("skills", [])),
        max_effort=int(data.get("max_effort", 6)),
    )


def _parse_robot(raw: dict[str, Any]) -> Robot:
    return Robot(
        name=str(raw.get("name", "Robot")),
        capabilities=frozenset(str(skill) for skill in raw.get("capabilities", [])),
        payload_kg=int(raw.get("payload_kg", 20)),
    )


def _parse_zone(raw: Any) -> Zone:
    data = _ensure_mapping(raw, "zone")
    position = data.get("position", [0, 0])
    if not isinstance(position, list | tuple) or len(position) != 2:
        raise ValueError(f"zone {data.get('id')} position must be [x, y]")
    return Zone(
        id=str(data["id"]),
        label=str(data.get("label", data["id"])),
        position=(float(position[0]), float(position[1])),
    )


def _parse_task(raw: Any) -> Task:
    data = _ensure_mapping(raw, "task")
    return Task(
        id=str(data["id"]),
        title=str(data["title"]),
        zone=str(data["zone"]),
        required_skill=str(data["required_skill"]),
        effort=int(data.get("effort", 1)),
        risk=str(data.get("risk", "low")),
        dependencies=tuple(str(item) for item in data.get("dependencies", [])),
    )


def _validate_references(zones: dict[str, Zone], tasks: tuple[Task, ...]) -> None:
    task_ids = {task.id for task in tasks}
    for task in tasks:
        if task.zone not in zones:
            raise ValueError(f"task {task.id} references unknown zone {task.zone}")
        for dependency in task.dependencies:
            if dependency not in task_ids:
                raise ValueError(f"task {task.id} references unknown dependency {dependency}")


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    return _ensure_mapping(raw.get(key), key)


def _sequence(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _ensure_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value
