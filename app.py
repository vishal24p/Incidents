# Import CLI, serialization, path, and typed-state helpers.
import argparse
import json
import operator
import os
from pathlib import Path
from typing import Annotated, TypedDict

# Require checkpoint values to serialize without silent fallbacks.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

# Reuse the tested specialists and fixture helpers.
from specialists import SPECIALISTS, alert_brief, build_model, load_json, run_specialist

DB_PATH = "triage_checkpoints.db"
SEVERITY_ORDER = ["SEV4", "SEV3", "SEV2", "SEV1"]
RUNBOOK_DIR = Path("runbooks")
RESTRICTED_TERMS = (
    "delete",
    "disable",
    "drop",
    "production",
    "restart",
    "rotate",
)

# Return one diagnosis with service-specific remediation actions.
ANALYSIS_SCHEMA = {
    "title": "IncidentPlan",
    "type": "object",
    "properties": {
        "severity": {
            "type": "string",
            "enum": ["SEV1", "SEV2", "SEV3", "SEV4"],
        },
        "summary": {"type": "string"},
        "likely_cause": {"type": "string"},
        "rationale": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["service", "action"],
            },
        },
    },
    "required": ["severity", "summary", "likely_cause", "rationale", "actions"],
}
# Constrain the Incident Commander's routing response.
ROUTER_SCHEMA = {
    "title": "TriageDecision",
    "type": "object",
    "properties": {
        "severity": {
            "type": "string",
            "enum": ["SEV1", "SEV2", "SEV3", "SEV4"],
        },
        "specialists": {
            "type": "array",
            "items": {"type": "string", "enum": list(SPECIALISTS)},
        },
        "reason": {"type": "string"},
    },
    "required": ["severity", "specialists", "reason"],
}


def highest_severity(current: str, new: str) -> str:
    # Keep the most urgent severity reported by any branch.
    if current not in SEVERITY_ORDER:
        return new
    return max(current, new, key=SEVERITY_ORDER.index)


# Store values shared across the incident workflow.
# Merge decisions returned by independent approval branches.
class IncidentState(TypedDict, total=False):
    incident_id: str
    incident: str
    alert: dict
    severity: Annotated[str, highest_severity]
    specialists: list[str]
    findings: Annotated[list[dict], operator.add]
    analysis: dict
    actions: list[dict]
    approvals: Annotated[list[dict], operator.add]
    status: str
    audit_log: list[str]
    runbook_path: str

# Give each parallel branch only the data it needs.
class SpecialistTask(TypedDict):
    specialist: str
    brief: str

# Carry one proposed action into its owning team's branch.
class ActionTask(TypedDict):
    incident_id: str
    team: str
    service: str
    action: str
    policy_reason: str


def build_analyst():
    # Apply the inherited analysis schema to the shared model.
    return build_model().with_structured_output(
        schema=ANALYSIS_SCHEMA,
        method="function_calling",
    )


def build_router():
    # Apply the routing schema to the Incident Commander.
    return build_model().with_structured_output(
        schema=ROUTER_SCHEMA,
        method="function_calling",
    )


def safe_incident_id(incident_id: str) -> str:
    safe_id = "".join(
        character
        for character in incident_id
        if character.isalnum() or character in {"-", "_"}
    )
    if not any(character.isalnum() for character in safe_id):
        raise ValueError("Incident ID must contain a letter or number.")
    return safe_id


def incident_brief(state: IncidentState) -> str:
    # Build the same evidence scope for every selected branch.
    return alert_brief(state["alert"]["id"])


def route_incident(state: IncidentState):
    # Ask the Incident Commander to classify and route the alert.
    decision = build_router().invoke(
        [
            (
                "system",
                "You are the incident commander. Set a first severity and choose "
                "which specialists investigate: logs, deploys, metrics. For SEV1 "
                "or SEV2, send every specialist. For SEV3 or SEV4, send only the "
                "one or two most likely to explain the symptom.",
            ),
            ("human", incident_brief(state)),
        ]
    )
    print(f"Router: {decision['severity']}, sending {', '.join(decision['specialists'])}")
    return {
        "severity": decision["severity"],
        "specialists": decision["specialists"],
        "status": "routed",
        "audit_log": state["audit_log"]
        + [f"Router chose {decision['specialists']}: {decision['reason']}"],
    }


def dispatch_specialists(state: IncidentState):
    # Create one parallel task for every selected specialist.
    brief = incident_brief(state)
    return [
        Send("investigate", {"specialist": name, "brief": brief})
        for name in state["specialists"]
    ]


def investigate(task: SpecialistTask):
    # Run one role inside its own parallel branch.
    name = task["specialist"]
    print(f"  {name} specialist investigating...")
    finding = run_specialist(name, task["brief"])
    print(f"  {name} specialist done: {finding['severity']}, {finding['suspected_cause']}")
    return {
        "findings": [{"specialist": name, **finding}],
        "severity": finding["severity"],
    }


def analyze_incident(state: IncidentState):
    # Turn the merged findings into service-specific fixes.
    analyst = build_analyst()
    findings = json.dumps(state["findings"], indent=2)
    analysis = analyst.invoke(
        [
            (
                "system",
                "You are an SRE incident analyst. Combine the specialists' "
                "findings into one diagnosis and name the most likely cause with "
                "its evidence. Propose one reversible remediation step for each "
                "service the evidence implicates: the alerting service and every "
                "dependency named in the findings' evidence, at most three. Each "
                "service's owning team must approve its own step. Keep every "
                "field concise.",
            ),
            ("human", f"{state['incident']}\n\nFindings:\n{findings}"),
        ]
    )
    return {
        "analysis": analysis,
        "severity": analysis["severity"],
        "status": "analyzed",
        "audit_log": state["audit_log"]
        + [f"Model combined {len(state['findings'])} findings into one analysis."],
    }


def policy_check(state: IncidentState):
    # Match every proposed service to its owning team.
    services = load_json("services.json")
    severity_requires_review = state["severity"] in {"SEV1", "SEV2"}
    actions, approvals = [], []
    for proposed in state["analysis"]["actions"]:
        service = services.get(proposed["service"], {"team": "unowned"})
        searchable_text = f"{state['incident']} {proposed['action']}".lower()
        term_requires_review = any(
            term in searchable_text for term in RESTRICTED_TERMS
        )
        requires_approval = severity_requires_review or term_requires_review
        action = {
            "incident_id": state["incident_id"],
            "team": service["team"],
            "service": proposed["service"],
            "action": proposed["action"],
            "policy_reason": (
                "High severity or restricted operational language detected."
                if requires_approval
                else "No high-risk policy rule matched."
            ),
        }
        actions.append(action | {"requires_approval": requires_approval})
                # Record low-risk actions without opening an interrupt.
        if not requires_approval:
            approvals.append(action | {"decision": "not required"})

    gated = sum(action["requires_approval"] for action in actions)
    return {
        "actions": actions,
        "approvals": approvals,
        "status": "policy_checked",
        "audit_log": state["audit_log"]
        + [f"Policy check: {gated} of {len(actions)} actions need team approval."],
    }


def policy_route(state: IncidentState):
    # Open one approval branch for every gated action.
    gated = [action for action in state["actions"] if action["requires_approval"]]
    if not gated:
        return "publish_runbook"
    return [
        Send(
            "approval_gate",
            {key: value for key, value in action.items() if key != "requires_approval"},
        )
        for action in gated
    ]


def approval_gate(task: ActionTask):
    # Pause this branch for the team that owns the service.
    decision = interrupt(
        {
            "question": f"Approve this action for team {task['team']}?",
            "incident_id": task["incident_id"],
            "team": task["team"],
            "service": task["service"],
            "action": task["action"],
            "policy_reason": task["policy_reason"],
            "allowed_decisions": ["approve", "reject"],
        }
    )
    if decision not in ("approve", "reject"):
        raise ValueError("Decision must be 'approve' or 'reject'.")
    return {"approvals": [task | {"decision": decision}]}


def publish_runbook(state: IncidentState):
    # Separate publishable actions from rejected actions.
    approvals = state.get("approvals", [])
    approved = [
        action
        for action in approvals
        if action["decision"] in ("approve", "not required")
    ]
    decisions = [
        f"Team {action['team']} decision for {action['service']}: {action['decision']}."
        for action in approvals
    ]
    if not approved:
        return {
            "status": "rejected_without_publish",
            "audit_log": state["audit_log"]
            + decisions
            + ["Every action was rejected. Publishing was skipped."],
        }
        # Build the Markdown sections from evidence and approved actions.
    RUNBOOK_DIR.mkdir(exist_ok=True)
    path = RUNBOOK_DIR / f"{safe_incident_id(state['incident_id'])}.md"
    analysis = state["analysis"]
    audit_log = state["audit_log"] + decisions + [f"Published runbook to {path}."]
    audit_lines = "\n".join(f"- {entry}" for entry in audit_log)
    finding_lines = "\n".join(
        f"- {finding['specialist']} ({finding['severity']}): {finding['summary']}"
        for finding in state["findings"]
    )
    action_lines = "\n".join(
        f"- {action['team']} / {action['service']}: {action['action']} "
        f"({'approved by team' if action['decision'] == 'approve' else 'approval not required'})"
        for action in approved
    )
        # Write the filtered actions and complete decision audit.
    path.write_text(
        "# Policy-Controlled Incident Runbook\n\n"
        f"Incident ID: {state['incident_id']}\n\n"
        f"Incident: {state['incident']}\n\n"
        f"Severity: {state['severity']}\n\n"
        f"Summary: {analysis['summary']}\n\n"
        f"Likely cause: {analysis['likely_cause']}\n\n"
        f"Rationale: {analysis['rationale']}\n\n"
        "Findings:\n"
        f"{finding_lines}\n\n"
        "Approved actions:\n"
        f"{action_lines}\n\n"
        "Audit log:\n"
        f"{audit_lines}\n",
        encoding="utf-8",
    )
    return {
        "status": "published",
        "runbook_path": str(path),
        "audit_log": audit_log,
    }


def build_graph(checkpointer: SqliteSaver):
    # Register routed investigations and team-owned approvals.
    builder = StateGraph(IncidentState)
    builder.add_node("route_incident", route_incident)
    builder.add_node("investigate", investigate)
    builder.add_node("analyze_incident", analyze_incident, defer=True)
    builder.add_node("policy_check", policy_check)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("publish_runbook", publish_runbook, defer=True)

    builder.add_edge(START, "route_incident")
    builder.add_conditional_edges(
        "route_incident", dispatch_specialists, ["investigate"]
    )
    builder.add_edge("investigate", "analyze_incident")
    builder.add_edge("analyze_incident", "policy_check")
    builder.add_conditional_edges(
        "policy_check", policy_route, ["approval_gate", "publish_runbook"]
    )
    builder.add_edge("approval_gate", "publish_runbook")
    builder.add_edge("publish_runbook", END)
    return builder.compile(checkpointer=checkpointer)


def display_stream(stream):
    # Show pending approvals or the final publication state.
    final_state = stream.output
    if stream.interrupted:
        display_pending(stream.interrupts)
        return

    print("WORKFLOW COMPLETE")
    print(f"Status: {final_state['status']}")
    print(f"Runbook: {final_state.get('runbook_path', 'not published')}")

def display_pending(interrupts):
    # List every team request that still needs a decision.
    print(f"WORKFLOW PAUSED: {len(interrupts)} team approvals pending")
    for pending in interrupts:
        request = pending.value
        print(f"- {request['team']} ({request['service']}): {request['action']}")

def display_failure(graph, config, error):
    # Read the durable snapshot after the branch failure.
    snapshot = graph.get_state(config)
    saved = [finding["specialist"] for finding in
             snapshot.values.get("findings", [])]

    # Separate completed branch output from work that must resume.
    print(f"WORKFLOW FAILED: {error}")
    print(f"Saved findings: {', '.join(saved) or 'none'}")
    print(f"Waiting to re-run: {', '.join(snapshot.next)}")


def parse_args():
    # Accept a team when resuming one pending decision.
    parser = argparse.ArgumentParser(
        description="Run a resumable incident-triage workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--thread", required=True)
    start_parser.add_argument("--alert", required=True)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--thread", required=True)
    resume_parser.add_argument(
        "--decision",
        choices=["approve", "reject"],
        help="Leave out to retry a failed run.",
    )
    resume_parser.add_argument("--team", help="The team whose action you decide.")

    args = parser.parse_args()
    if args.command == "resume" and args.decision and not args.team:
        parser.error("--decision needs --team.")
    return args


def main():
    # Reopen the same checkpoint thread for starts and resumes.
    args = parse_args()
    config = {"configurable": {"thread_id": args.thread}}

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer)

        if args.command == "start":
            # Load the selected fixture into shared graph state.
            alerts = {alert["id"]: alert for alert in load_json("alerts.json")}
            alert = alerts[args.alert]
            graph_input = {
                "incident_id": args.thread,
                "incident": f"{alert['title']}: {alert['description']}",
                "alert": alert,
                "status": "new",
                "audit_log": [],
            }
        elif args.decision:
            pending = graph.get_state(config).interrupts
            team_requests = {
                request.id: args.decision
                for request in pending
                if request.value["team"] == args.team
            }
            if not team_requests:
                teams = sorted({request.value["team"] for request in pending})
                print(
                    f"No pending approval for team {args.team}. "
                    f"Waiting on: {', '.join(teams) or 'nobody'}"
                )
                return
            graph_input = Command(resume=team_requests)
            print(f"Recorded {args.decision} for {args.team}.")
        else:
            graph_input = None

        try:
            stream = graph.stream_events(
                graph_input, config=config, durability="sync", version="v3"
            )
            display_stream(stream)
        except Exception as error:
            # Report the checkpoint instead of hiding it behind a traceback.
            display_failure(graph, config, error)

if __name__ == "__main__":
    main()
