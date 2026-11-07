import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import app
import llm
import specialists


class WorkflowTests(unittest.TestCase):
    analysis = {
        "severity": "SEV3",
        "summary": "Elevated queue latency.",
        "likely_cause": "Queue saturation.",
        "rationale": "Read-only investigation.",
        "actions": [{"service": "checkout-api", "action": "Inspect queue metrics."}],
    }

    def test_specialists_reuse_shared_factory(self):
        self.assertIs(specialists.build_model, llm.build_model)

    def test_highest_severity_reducer(self):
        self.assertEqual(app.highest_severity("SEV3", "SEV1"), "SEV1")
        self.assertEqual(app.highest_severity("SEV2", "SEV4"), "SEV2")
        self.assertEqual(app.highest_severity("", "SEV2"), "SEV2")

    def test_analyst_uses_tool_calling_and_returns_schema_dict(self):
        model = ChatOpenAI(model="test-model", api_key="offline")
        response = ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="",
            tool_calls=[{
                "name": app.ANALYSIS_SCHEMA["title"],
                "args": self.analysis,
                "id": "analysis-call",
                "type": "tool_call",
            }],
        ))])
        with patch("app.build_model", return_value=model), patch.object(
            ChatOpenAI, "_generate", return_value=response
        ) as generate:
            result = app.build_analyst().invoke("Inspect the incident.")
        self.assertEqual(result, self.analysis)
        request = generate.call_args.kwargs
        self.assertEqual(request["tools"][0]["function"]["parameters"], {
            key: value for key, value in app.ANALYSIS_SCHEMA.items() if key != "title"
        })
        self.assertEqual(request["tool_choice"]["function"]["name"], "IncidentPlan")
        self.assertNotIn("response_format", request)

    def test_router_uses_tool_calling_and_returns_schema_dict(self):
        decision = {
            "severity": "SEV2",
            "specialists": ["logs", "deploys", "metrics"],
            "reason": "High impact requires broad investigation.",
        }
        model = ChatOpenAI(model="test-model", api_key="offline")
        response = ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="",
            tool_calls=[{
                "name": app.ROUTER_SCHEMA["title"],
                "args": decision,
                "id": "router-call",
                "type": "tool_call",
            }],
        ))])
        with patch("app.build_model", return_value=model), patch.object(
            ChatOpenAI, "_generate", return_value=response
        ) as generate:
            result = app.build_router().invoke("Route the incident.")
        self.assertEqual(result, decision)
        request = generate.call_args.kwargs
        self.assertEqual(request["tools"][0]["function"]["parameters"], {
            key: value for key, value in app.ROUTER_SCHEMA.items() if key != "title"
        })
        self.assertEqual(request["tool_choice"]["function"]["name"], "TriageDecision")
        self.assertNotIn("response_format", request)

    def test_low_risk_publishes_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            runbooks = Path(directory) / "runbooks"
            analyst = Mock()
            analyst.invoke.return_value = self.analysis
            router = Mock()
            router.invoke.return_value = {
                "severity": "SEV3",
                "specialists": ["logs"],
                "reason": "Low-risk performance issue.",
            }
            finding = {
                "summary": "Queue latency is elevated.",
                "evidence": ["Queue latency increased."],
                "suspected_cause": "Read-only investigation needed.",
                "severity": "SEV3",
            }
            with patch("app.build_router", return_value=router), patch(
                "app.run_specialist", return_value=finding
            ), patch("app.alert_brief", return_value="Queue latency alert."), patch(
                "app.build_analyst", return_value=analyst
            ), patch.object(app, "RUNBOOK_DIR", runbooks), SqliteSaver.from_conn_string(
                str(Path(directory) / "checkpoints.db")
            ) as saver:
                graph = app.build_graph(saver)
                result = graph.invoke({
                    "incident_id": "low-risk",
                    "incident": "Queue latency increased.",
                    "alert": {"id": "INC-low-risk"},
                    "audit_log": [],
                }, {"configurable": {"thread_id": "low-risk"}})
            self.assertEqual(result["status"], "published")
            self.assertFalse(result["actions"][0]["requires_approval"])
            self.assertEqual(result["approvals"][0]["decision"], "not required")
            self.assertEqual(result["analysis"], self.analysis)
            content = Path(result["runbook_path"]).read_text(encoding="utf-8")
            self.assertIn("payments / checkout-api: Inspect queue metrics. (approval not required)", content)
            analyst.invoke.assert_called_once()

    def test_high_risk_pauses_and_resumes_from_sqlite(self):
        for decision in ("approve", "reject"):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                runbooks = Path(directory) / "runbooks"
                database = str(Path(directory) / "checkpoints.db")
                config = {"configurable": {"thread_id": decision}}
                analyst = Mock()
                analyst.invoke.return_value = {
                    **self.analysis, "severity": "SEV1"
                }
                router = Mock()
                router.invoke.return_value = {
                    "severity": "SEV2",
                    "specialists": ["logs"],
                    "reason": "High-risk incident needs investigation.",
                }
                finding = {
                    "summary": "Checkout errors are increasing.",
                    "evidence": ["HTTP 500 errors observed."],
                    "suspected_cause": "Connection pool exhaustion.",
                    "severity": "SEV2",
                }
                with patch("app.build_router", return_value=router), patch(
                    "app.run_specialist", return_value=finding
                ), patch("app.alert_brief", return_value="Checkout alert."), patch(
                    "app.build_analyst", return_value=analyst
                ), patch.object(app, "RUNBOOK_DIR", runbooks):
                    with SqliteSaver.from_conn_string(database) as saver:
                        result = app.build_graph(saver).invoke({
                            "incident_id": decision,
                            "incident": "Queue latency increased.",
                            "alert": {"id": f"INC-{decision}"},
                            "audit_log": [],
                        }, config)
                    self.assertTrue(result["actions"][0]["requires_approval"])
                    pending, = result["__interrupt__"]
                    self.assertEqual(pending.value["team"], "payments")
                    self.assertEqual(pending.value["allowed_decisions"], ["approve", "reject"])
                    self.assertFalse(runbooks.exists())
                    with SqliteSaver.from_conn_string(database) as saver:
                        graph = app.build_graph(saver)
                        self.assertEqual(graph.get_state(config).next, ("approval_gate",))
                        result = graph.invoke(Command(resume={pending.id: decision}), config)
                    analyst.invoke.assert_called_once()
                self.assertEqual(result["approvals"][0]["decision"], decision)
                if decision == "approve":
                    self.assertEqual(result["status"], "published")
                    content = Path(result["runbook_path"]).read_text(encoding="utf-8")
                    self.assertIn("payments / checkout-api: Inspect queue metrics. (approved by team)", content)
                else:
                    self.assertEqual(result["status"], "rejected_without_publish")
                    self.assertNotIn("runbook_path", result)
                    self.assertFalse(runbooks.exists())
                    self.assertIn("Team payments decision for checkout-api: reject.", result["audit_log"])


if __name__ == "__main__":
    unittest.main()
