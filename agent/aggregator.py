"""Result Aggregation for single-tool and multi-tool specialist executions.

Normalizes outputs, combines evidence collections, aggregates generated artifacts,
and compiles the operational execution trace.
"""

from __future__ import annotations

from typing import List, Optional

from core.logging import get_logger
from core.schemas import (
    AgentDecision,
    Artifact,
    Evidence,
    ExecutionStage,
    ExecutionTraceEntry,
    QueryResponse,
    SatQueryErrorDetail,
    TaskIntent,
    TaskPlan,
    TaskType,
    ToolResult,
    ToolStatus,
)

logger = get_logger("aggregator")


class ResultAggregator:
    """Standardizes and synthesizes multiple specialist tool results into a unified response."""

    @classmethod
    def aggregate(
        cls,
        request_id: str,
        query: str,
        resolved_task: TaskType,
        tool_results: List[ToolResult],
        system_trace_entries: Optional[List[ExecutionTraceEntry]] = None,
        errors: Optional[List[SatQueryErrorDetail]] = None,
        task_intent: Optional[TaskIntent] = None,
        task_plan: Optional[TaskPlan] = None,
        agent_decision: Optional[AgentDecision] = None,
        selected_tools: Optional[List[str]] = None,
    ) -> QueryResponse:
        """Aggregate one or more ToolResults into a final QueryResponse."""
        system_trace = list(system_trace_entries or [])
        errs = list(errors or [])
        sel_tools = list(selected_tools or [r.model_info.get("name", "specialist") for r in tool_results])

        # If no tool results and errors exist, build error response
        if not tool_results:
            return QueryResponse(
                request_id=request_id,
                query=query,
                resolved_task=resolved_task,
                status=ToolStatus.FAILED,
                answer="Analysis could not be completed due to errors.",
                confidence=None,
                evidence=[],
                artifacts=[],
                execution_trace=system_trace,
                task_intent=task_intent,
                task_plan=task_plan,
                agent_decision=agent_decision,
                selected_tools=sel_tools,
                errors=errs,
                metadata={},
            )

        # Single Tool Result case (standard flow)
        if len(tool_results) == 1:
            res = tool_results[0]
            # Merge tool trace entries with system trace
            all_trace = system_trace + res.execution_trace
            all_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.RESULT_AGGREGATED,
                    component="ResultAggregator",
                    status="COMPLETED",
                    details={"tool_count": 1, "evidence_count": len(res.evidence)},
                )
            )

            return QueryResponse(
                request_id=request_id,
                query=query,
                resolved_task=resolved_task,
                status=res.status,
                answer=res.answer,
                confidence=res.confidence,
                evidence=res.evidence,
                artifacts=res.artifacts,
                execution_trace=all_trace,
                task_intent=task_intent,
                task_plan=task_plan,
                agent_decision=agent_decision,
                selected_tools=sel_tools,
                errors=errs,
                metadata={
                    "model_info": res.model_info,
                    "parameters": res.parameters,
                    **res.metadata,
                },
            )

        # Multi-Tool Sequential Results case
        combined_answers = []
        all_evidence: List[Evidence] = []
        all_artifacts: List[Artifact] = []
        confidences = []
        overall_status = ToolStatus.SUCCESS

        for idx, res in enumerate(tool_results):
            combined_answers.append(f"[Step {idx + 1} - {res.task.value}]: {res.answer}")
            all_evidence.extend(res.evidence)
            all_artifacts.extend(res.artifacts)
            if res.confidence is not None:
                confidences.append(res.confidence)
            if res.status != ToolStatus.SUCCESS:
                overall_status = res.status
            system_trace.extend(res.execution_trace)

        final_answer = "\n\n".join(combined_answers)
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None

        system_trace.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.RESULT_AGGREGATED,
                component="ResultAggregator",
                status="COMPLETED",
                details={
                    "tool_count": len(tool_results),
                    "evidence_count": len(all_evidence),
                    "artifact_count": len(all_artifacts),
                },
            )
        )

        return QueryResponse(
            request_id=request_id,
            query=query,
            resolved_task=resolved_task,
            status=overall_status,
            answer=final_answer,
            confidence=avg_confidence,
            evidence=all_evidence,
            artifacts=all_artifacts,
            execution_trace=system_trace,
            task_intent=task_intent,
            task_plan=task_plan,
            agent_decision=agent_decision,
            selected_tools=sel_tools,
            errors=errs,
            metadata={"workflow_steps": len(tool_results)},
        )
