"""ALFRED's goal conditions, evaluated on the text environment's facts.

A per-step label on ALFWorld cannot be the outcome check. ``won`` is false at
every non-terminal step, so grading a prefix by it labels every prefix of a
successful episode a failure until the last one, and the label degenerates into
"has the episode finished" -- which says nothing about whether the agent is on
track. On code the label comes from a verifier applied to an artefact; here
there is no artefact.

The graded quantity ALFRED uses instead is the fraction of the task's goal
conditions satisfied, and it is defined at any state. The embodied environment
computes it (``alfred_thor_env.py``: ``get_goal_conditions_met``); the text
environment does not, and ``AlfredTWEnv.init_env`` does not even request the
logic it would need. But the logic is there once asked for: ``facts`` is every
proposition true in the current state, and ``traj_data.json`` carries the
quest's parameters. ALFRED's own counts are then reproducible predicate by
predicate -- the task classes in ``alfworld/env/tasks.py`` read a THOR state,
and each of their checks has an exact counterpart here:

    object of class X in a receptacle of class Y   inreceptacle(o, r)
    object was heated / cooled / cleaned           ishot / iscool / isclean
    object is held                                 holds(agent, o)
    lamp is on                                     istoggled(o)
    object is sliced                               issliced(o)
    object inside a movable receptacle             inreceptacleobject(o, m)

The three treatment predicates mean what ALFRED means by them: the domain
annotates ``isClean`` as "true if the object has been clean in sink" and heating
asserts ``isHot`` while retracting ``isCool``, so they record that the treatment
happened rather than that it still holds -- the same semantics as ALFRED's
``heated_objects`` / ``cooled_objects`` / ``cleaned_objects`` sets.

One check does not survive the move to text: ``look_at_obj_in_light`` wants the
lamp toggled *and visible*, and visibility has no proposition. We score the
toggle alone, which makes that one condition easier to satisfy than ALFRED's.

Nothing here is trusted on faith: ``--validate`` replays each episode to its
last step and compares "all conditions satisfied" against the recorded outcome.
"""
from __future__ import annotations

from collections import defaultdict

#: the treatment each task demands of its object
TREATMENT = {"pick_heat_then_place_in_recep": "ishot",
             "pick_cool_then_place_in_recep": "iscool",
             "pick_clean_then_place_in_recep": "isclean"}


def index(facts) -> dict[str, set[tuple[str, ...]]]:
    """Propositions by predicate name, arguments stripped of their padding."""
    out: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for fact in facts:
        out[fact.name.lower()].add(tuple(str(a.name).strip() for a in fact.arguments))
    return out


def typed(idx: dict, predicate: str, klass: str) -> set[str]:
    """The entities whose declared type is this ALFRED class."""
    if not klass:
        return set()
    want = klass.strip().lower() + "type"
    return {args[0] for args in idx.get(predicate, ()) if len(args) > 1
            and args[1].lower() == want}


def goal_conditions(task_type: str, params: dict, facts) -> tuple[int, int] | None:
    """How many of the quest's conditions hold, and how many there are.

    Returns None for a task type we do not model, rather than a number that
    looks like a measurement.
    """
    idx = index(facts)
    objects = typed(idx, "objecttype", params.get("object_target") or "")
    parents = typed(idx, "receptacletype", params.get("parent_target") or "")
    in_receptacle = idx.get("inreceptacle", set())
    in_object = idx.get("inreceptacleobject", set())
    held = {args[1] for args in idx.get("holds", ()) if len(args) > 1}
    sliced = {o for o in objects if (o,) in idx.get("issliced", set())}
    in_place = {o for o in objects if any((o, r) in in_receptacle for r in parents)}

    satisfied = total = 0

    def require(condition: bool) -> None:
        nonlocal satisfied, total
        total += 1
        satisfied += int(bool(condition))

    if task_type == "pick_and_place_simple":
        require(bool(in_place))
    elif task_type == "pick_two_obj_and_place":
        best = max((sum(1 for o in objects if (o, r) in in_receptacle) for r in parents),
                   default=0)
        total += 2
        satisfied += min(best, 2)
    elif task_type == "look_at_obj_in_light":
        require(bool(objects & held))
        toggles = typed(idx, "objecttype", params.get("toggle_target") or "")
        # ALFRED also wants the lamp visible; nothing in the logic says so
        require(any((t,) in idx.get("istoggled", set()) for t in toggles))
    elif task_type in TREATMENT:
        treated = {o for o in objects if (o,) in idx.get(TREATMENT[task_type], set())}
        require(bool(in_place))
        require(bool(treated))
        require(bool(in_place & treated))
    elif task_type == "pick_and_place_with_movable_recep":
        movables = typed(idx, "objecttype", params.get("mrecep_target") or "")
        inside = {m for m in movables if any((m, r) in in_receptacle for r in parents)}
        require(any((o, m) in in_object for o in objects for m in movables))
        require(bool(inside))
        require(any((o, m) in in_object for o in objects for m in inside))
    else:
        return None

    if params.get("object_sliced"):
        # ALFRED counts the slicing separately, twice when two objects are wanted
        if task_type == "pick_two_obj_and_place":
            total += 2
            satisfied += min(len(sliced), 2)
        else:
            require(bool(sliced))

    return satisfied, total
