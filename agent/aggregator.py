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
        agg_metadata: dict = {"workflow_steps": len(tool_results)}

        is_change_vqa_workflow = (resolved_task == TaskType.CHANGE_VQA)

        for idx, res in enumerate(tool_results):
            # Human-readable stage labeling for composed Change-VQA
            if is_change_vqa_workflow:
                if idx == 0 and "change" in res.task.value.lower():
                    stage_title = "Step 1 [Change Detection Stage - TinyCD]"
                elif idx == 1:
                    stage_title = "Step 2 [Semantic Interpretation Stage - VLM]"
                else:
                    stage_title = f"Step {idx + 1} - {res.task.value}"
            else:
                stage_title = f"Step {idx + 1} - {res.task.value}"

            combined_answers.append(f"[{stage_title}]:\n{res.answer}")
            all_evidence.extend(res.evidence)
            all_artifacts.extend(res.artifacts)
            if res.confidence is not None:
                confidences.append(res.confidence)
            if res.status != ToolStatus.SUCCESS:
                overall_status = res.status

            # Extract provenance & metadata
            if res.metadata:
                if "tinycd_provenance" in res.metadata:
                    agg_metadata["tinycd_provenance"] = res.metadata["tinycd_provenance"]
                if "changed_region_evidence" in res.metadata:
                    agg_metadata["changed_region_evidence"] = res.metadata["changed_region_evidence"]
                if "vlm_provenance" in res.metadata:
                    agg_metadata["vlm_provenance"] = res.metadata["vlm_provenance"]
                if "change_vqa_status" in res.metadata:
                    agg_metadata["change_vqa_status"] = res.metadata["change_vqa_status"]
                if "tinycd_detection" in res.metadata:
                    agg_metadata["tinycd_detection"] = res.metadata["tinycd_detection"]
                if "semantic_vlm_interpretation" in res.metadata:
                    agg_metadata["semantic_vlm_interpretation"] = res.metadata["semantic_vlm_interpretation"]
                if "evidence_package" in res.metadata:
                    agg_metadata["evidence_package"] = res.metadata["evidence_package"]
                if "evidence_artifact_ids" in res.metadata:
                    agg_metadata["evidence_artifact_ids"] = res.metadata["evidence_artifact_ids"]

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
                    "is_change_vqa": is_change_vqa_workflow,
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
            metadata=agg_metadata,
        )
