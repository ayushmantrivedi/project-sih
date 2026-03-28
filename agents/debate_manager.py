from utils import get_llama_client, get_logger
import json

logger = get_logger("debate_manager")

class DebateManager:
    """
    Orchestrates the 3-turn 'Chief of Medicine' debate loop.
    Ensures accuracy, filters hallucinations, and maintains query focus.
    """
    def __init__(self, model="phi3:mini", max_turns=3):
        self.llama = get_llama_client()
        self.model = model
        self.max_turns = max_turns

    def run_debate(self, query, rag_evidence, user_history):
        """
        Runs the full debate loop and returns the final synthesized response.
        """
        # Turn 1: The Clinician (Proposer)
        logger.info("Debate Turn 1: The Clinician")
        initial_proposal = self._get_clinician_proposal(query, rag_evidence, user_history)
        
        # Turn 2: The Medical Auditor (Critic)
        logger.info("Debate Turn 2: The Medical Auditor")
        audit_report = self._get_auditor_critique(query, initial_proposal, rag_evidence)
        
        # Turn 3: The Chief of Medicine (Judge)
        logger.info("Debate Turn 3: The Chief of Medicine")
        final_verdict = self._get_judge_final_decision(query, initial_proposal, audit_report, rag_evidence)
        
        return {
            "final_answer": final_verdict,
            "debate_log": {
                "initial_proposal": initial_proposal,
                "audit_report": audit_report
            }
        }

    def _get_clinician_proposal(self, query, evidence, history):
        prompt = f"""
        Role: Senior Clinician. 
        Goal: Provide a detailed clinical assessment based on the Evidence and User History.
        
        Query: {query}
        Evidence: {evidence}
        User History: {history}
        
        Guidelines:
        1. Base your answer strictly on the evidence and history provided.
        2. If evidence is missing, state what is unknown.
        3. ALWAYS cite the Source ID [...] from the evidence.
        4. Focus exclusively on the query.
        """
        return self.llama.generate_reasoning(prompt, "Symptom Analysis")

    def _get_auditor_critique(self, query, proposal, evidence):
        prompt = f"""
        Role: Medical Auditor / Fact-Checker.
        Goal: Identify hallucinations, factual errors, or over-confident claims in the Clinician's proposal.
        
        Original Query: {query}
        Clinician's Proposal: {proposal}
        Reference Evidence: {evidence}
        
        Audit Task:
        1. Compare every clinical claim in the proposal against the reference evidence.
        2. List "Points of Hallucination": Any claim NOT supported by the evidence.
        3. List "Points of Contradiction": Any claim that directly opposes the evidence.
        4. Do NOT verify using your internal knowledge; ONLY use the provided Evidence.
        """
        return self.llama.generate_reasoning(prompt, "Hallucination Audit")

    def _get_judge_final_decision(self, query, proposal, audit, evidence):
        prompt = f"""
        Role: Chief of Medicine.
        Goal: Summarize the final medical guidance, correcting any errors identified by the Auditor.
        
        Original Query: {query}
        Original Proposal: {proposal}
        Audit Log: {audit}
        Reference Evidence: {evidence}
        
        Synthesis Task:
        1. Incorporate valid points from the Clinician while fixing all hallucinations flagged by the Auditor.
        2. Ensure the response is focused, professional, and directly answers the user's query.
        3. If the situation is serious, emphasize immediate safety steps for the user.
        4. STOP the debate here. Do not loop.
        """
        return self.llama.generate_reasoning(prompt, "Final Verdict")

# Singleton
_debate_manager = None

def get_debate_manager():
    global _debate_manager
    if _debate_manager is None:
        _debate_manager = DebateManager()
    return _debate_manager
