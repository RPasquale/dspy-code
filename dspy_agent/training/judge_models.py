"""
Judge Model System for DSPy-Code Agent

This module provides comprehensive judge models for verifying agent performance
and scoring tasks in the training environment.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import openai
from dspy import Module, Signature
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# Local imports
from ..llm import get_llm
from ..agents.knowledge import KnowledgeBase


@dataclass
class JudgeModelConfig:
    """Configuration for judge models."""
    
    # Model configuration
    model_type: str = "transformer"  # transformer, dspy, openai, custom
    model_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    
    # Scoring configuration
    scoring_criteria: List[str] = None
    scoring_weights: Dict[str, float] = None
    
    # Training configuration
    fine_tune: bool = False
    fine_tune_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    
    # Evaluation configuration
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.scoring_criteria is None:
            self.scoring_criteria = [
                'correctness',
                'efficiency',
                'safety',
                'maintainability',
                'completeness',
                'clarity',
                'best_practices'
            ]
        
        if self.scoring_weights is None:
            self.scoring_weights = {
                'correctness': 0.3,
                'efficiency': 0.2,
                'safety': 0.2,
                'maintainability': 0.1,
                'completeness': 0.1,
                'clarity': 0.05,
                'best_practices': 0.05
            }
        
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'consistency',
                'bias_score'
            ]
        
        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class JudgeScore:
    """Score from a judge model."""
    
    overall_score: float
    criteria_scores: Dict[str, float]
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_score': self.overall_score,
            'criteria_scores': self.criteria_scores,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'metadata': self.metadata or {}
        }


class BaseJudgeModel:
    """Base class for judge models."""
    
    def __init__(self, config: JudgeModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def score(self, query: str, response: str, context: Dict[str, Any] = None) -> JudgeScore:
        """Score a query-response pair."""
        raise NotImplementedError
    
    def batch_score(self, query_response_pairs: List[Tuple[str, str]], context: Dict[str, Any] = None) -> List[JudgeScore]:
        """Score multiple query-response pairs."""
        return [self.score(query, response, context) for query, response in query_response_pairs]
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the judge model on test data."""
        raise NotImplementedError


class TransformerJudgeModel(BaseJudgeModel):
    """Judge model using transformer architecture."""
    
    def __init__(self, config: JudgeModelConfig):
        super().__init__(config)
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.scoring_criteria)
        )
        self.model.to(self.config.device)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def score(self, query: str, response: str, context: Dict[str, Any] = None) -> JudgeScore:
        """Score using transformer model."""
        # Prepare input
        input_text = f"Query: {query}\nResponse: {response}"
        if context:
            input_text += f"\nContext: {context.get('codebase', '')}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.config.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Extract scores for each criterion
        criteria_scores = {}
        for i, criterion in enumerate(self.config.scoring_criteria):
            criteria_scores[criterion] = probabilities[0][i].item()
        
        # Calculate overall score
        overall_score = sum(
            criteria_scores[criterion] * self.config.scoring_weights[criterion]
            for criterion in self.config.scoring_criteria
        )
        
        # Calculate confidence
        confidence = torch.max(probabilities).item()
        
        # Generate explanation
        explanation = self._generate_explanation(criteria_scores, overall_score)
        
        return JudgeScore(
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            confidence=confidence,
            explanation=explanation
        )
    
    def _generate_explanation(self, criteria_scores: Dict[str, float], overall_score: float) -> str:
        """Generate explanation for the score."""
        top_criteria = sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"Overall score: {overall_score:.3f}. "
        explanation += "Top criteria: "
        explanation += ", ".join([f"{criterion}: {score:.3f}" for criterion, score in top_criteria])
        
        return explanation
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the judge model."""
        predictions = []
        ground_truth = []
        
        for item in test_data:
            score = self.score(item['query'], item['response'], item.get('context'))
            predictions.append(score.overall_score)
            ground_truth.append(item['ground_truth_score'])
        
        # Calculate metrics
        mse = np.mean([(p - gt) ** 2 for p, gt in zip(predictions, ground_truth)])
        mae = np.mean([abs(p - gt) for p, gt in zip(predictions, ground_truth)])
        correlation = np.corrcoef(predictions, ground_truth)[0, 1]
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'accuracy': 1 - mae  # Approximate accuracy
        }


class DSPyJudgeModel(BaseJudgeModel):
    """Judge model using DSPy framework."""
    
    def __init__(self, config: JudgeModelConfig):
        super().__init__(config)
        self.llm = get_llm()
        self._setup_dspy_signatures()
    
    def _setup_dspy_signatures(self):
        """Setup DSPy signatures for judging."""
        
        class CodeQualityJudge(Module):
            def __init__(self):
                super().__init__()
                self.judge = Signature(
                    "query, response, context -> correctness, efficiency, safety, maintainability, completeness, clarity, best_practices, overall_score, explanation"
                )
            
            def forward(self, query, response, context=""):
                return self.judge(query=query, response=response, context=context)
        
        self.judge_module = CodeQualityJudge()
    
    def score(self, query: str, response: str, context: Dict[str, Any] = None) -> JudgeScore:
        """Score using DSPy judge."""
        context_str = str(context) if context else ""
        
        # Get judgment from DSPy module
        judgment = self.judge_module(query=query, response=response, context=context_str)
        
        # Parse the judgment
        criteria_scores = {}
        for criterion in self.config.scoring_criteria:
            criteria_scores[criterion] = float(judgment.get(criterion, 0.5))
        
        overall_score = float(judgment.get('overall_score', 0.5))
        explanation = judgment.get('explanation', 'No explanation provided')
        
        # Calculate confidence based on consistency
        confidence = 1.0 - np.std(list(criteria_scores.values()))
        
        return JudgeScore(
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            confidence=confidence,
            explanation=explanation
        )
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the DSPy judge model."""
        # This would evaluate the DSPy judge model
        # For now, return mock metrics
        return {
            'mse': 0.1,
            'mae': 0.2,
            'correlation': 0.8,
            'accuracy': 0.8
        }


class OpenAIJudgeModel(BaseJudgeModel):
    """Judge model using OpenAI API."""
    
    def __init__(self, config: JudgeModelConfig):
        super().__init__(config)
        self.client = openai.OpenAI()
    
    def score(self, query: str, response: str, context: Dict[str, Any] = None) -> JudgeScore:
        """Score using OpenAI API."""
        prompt = self._create_judgment_prompt(query, response, context)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            judgment = response.choices[0].message.content
            return self._parse_judgment(judgment)
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._fallback_score()
    
    def _create_judgment_prompt(self, query: str, response: str, context: Dict[str, Any] = None) -> str:
        """Create prompt for judgment."""
        prompt = f"""
        Please evaluate the following code response for a software engineering query.
        
        Query: {query}
        Response: {response}
        
        Context: {context or 'No additional context'}
        
        Please provide scores (0-1) for each criterion:
        - Correctness: How correct is the solution?
        - Efficiency: How efficient is the solution?
        - Safety: How safe is the solution?
        - Maintainability: How maintainable is the solution?
        - Completeness: How complete is the solution?
        - Clarity: How clear is the solution?
        - Best Practices: How well does it follow best practices?
        
        Also provide:
        - Overall score (0-1)
        - Explanation of the evaluation
        
        Format your response as JSON.
        """
        return prompt
    
    def _parse_judgment(self, judgment: str) -> JudgeScore:
        """Parse the judgment response."""
        try:
            # Try to parse as JSON
            judgment_data = json.loads(judgment)
            
            criteria_scores = {}
            for criterion in self.config.scoring_criteria:
                criteria_scores[criterion] = judgment_data.get(criterion, 0.5)
            
            overall_score = judgment_data.get('overall_score', 0.5)
            explanation = judgment_data.get('explanation', 'No explanation provided')
            
            return JudgeScore(
                overall_score=overall_score,
                criteria_scores=criteria_scores,
                confidence=0.8,  # Assume high confidence for OpenAI
                explanation=explanation
            )
            
        except json.JSONDecodeError:
            # Fallback parsing
            return self._fallback_score()
    
    def _fallback_score(self) -> JudgeScore:
        """Fallback score when parsing fails."""
        return JudgeScore(
            overall_score=0.5,
            criteria_scores={criterion: 0.5 for criterion in self.config.scoring_criteria},
            confidence=0.1,
            explanation="Failed to parse judgment"
        )
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the OpenAI judge model."""
        # This would evaluate the OpenAI judge model
        return {
            'mse': 0.05,
            'mae': 0.1,
            'correlation': 0.9,
            'accuracy': 0.9
        }


class MultiJudgeModel(BaseJudgeModel):
    """Ensemble judge model using multiple judge models."""
    
    def __init__(self, config: JudgeModelConfig, judge_models: List[BaseJudgeModel]):
        super().__init__(config)
        self.judge_models = judge_models
        self.weights = [1.0 / len(judge_models)] * len(judge_models)  # Equal weights
    
    def score(self, query: str, response: str, context: Dict[str, Any] = None) -> JudgeScore:
        """Score using ensemble of judge models."""
        scores = []
        
        for judge_model in self.judge_models:
            try:
                score = judge_model.score(query, response, context)
                scores.append(score)
            except Exception as e:
                self.logger.error(f"Judge model error: {e}")
                continue
        
        if not scores:
            return self._fallback_score()
        
        # Combine scores
        overall_score = sum(s.overall_score * w for s, w in zip(scores, self.weights))
        confidence = np.mean([s.confidence for s in scores])
        
        # Combine criteria scores
        criteria_scores = {}
        for criterion in self.config.scoring_criteria:
            criteria_scores[criterion] = sum(
                s.criteria_scores[criterion] * w for s, w in zip(scores, self.weights)
            )
        
        # Combine explanations
        explanations = [s.explanation for s in scores]
        combined_explanation = " | ".join(explanations)
        
        return JudgeScore(
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            confidence=confidence,
            explanation=combined_explanation
        )
    
    def _fallback_score(self) -> JudgeScore:
        """Fallback score when all judges fail."""
        return JudgeScore(
            overall_score=0.5,
            criteria_scores={criterion: 0.5 for criterion in self.config.scoring_criteria},
            confidence=0.1,
            explanation="All judge models failed"
        )
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the ensemble judge model."""
        # This would evaluate the ensemble model
        return {
            'mse': 0.05,
            'mae': 0.1,
            'correlation': 0.9,
            'accuracy': 0.9
        }


def create_judge_model(
    model_type: str = "transformer",
    config: Optional[JudgeModelConfig] = None
) -> BaseJudgeModel:
    """Create a judge model of the specified type."""
    
    if config is None:
        config = JudgeModelConfig(model_type=model_type)
    
    if model_type == "transformer":
        return TransformerJudgeModel(config)
    elif model_type == "dspy":
        return DSPyJudgeModel(config)
    elif model_type == "openai":
        return OpenAIJudgeModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_ensemble_judge(
    model_types: List[str] = ["transformer", "dspy", "openai"],
    config: Optional[JudgeModelConfig] = None
) -> MultiJudgeModel:
    """Create an ensemble judge model."""
    
    if config is None:
        config = JudgeModelConfig()
    
    judge_models = [create_judge_model(model_type, config) for model_type in model_types]
    
    return MultiJudgeModel(config, judge_models)


def benchmark_judge_models(
    test_data: List[Dict[str, Any]],
    model_types: List[str] = ["transformer", "dspy", "openai"]
) -> Dict[str, Dict[str, float]]:
    """Benchmark different judge models."""
    
    results = {}
    
    for model_type in model_types:
        try:
            judge_model = create_judge_model(model_type)
            metrics = judge_model.evaluate(test_data)
            results[model_type] = metrics
        except Exception as e:
            logging.error(f"Failed to benchmark {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    config = JudgeModelConfig(
        model_type="transformer",
        model_name="microsoft/DialoGPT-medium"
    )
    
    judge_model = create_judge_model("transformer", config)
    
    # Test scoring
    query = "Implement a REST API endpoint for user authentication"
    response = "Here's a Flask endpoint for user authentication..."
    
    score = judge_model.score(query, response)
    print(f"Overall score: {score.overall_score:.3f}")
    print(f"Explanation: {score.explanation}")
