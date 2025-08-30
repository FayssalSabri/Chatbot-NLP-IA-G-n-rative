import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from business_rules.rules_engine import (
    BusinessRulesEngine, BusinessRule, Priority, ActionType,
    SentimentCondition, RatingCondition, KeywordCondition, AspectCondition
)

class TestBusinessRulesEngine:
    
    @pytest.fixture
    def rules_engine(self):
        # Create engine without default rules for testing
        engine = BusinessRulesEngine()
        engine.rules = []  # Clear default rules
        return engine
    
    @pytest.fixture
    def sample_context(self):
        return {
            'text': 'le service était mauvais et le produit défectueux',
            'rating': 2,
            'sentiment_polarity': -0.8,
            'sentiment_subjectivity': 0.9,
            'aspects': ['service', 'produit'],
            'entities': []
        }
    
    def test_sentiment_condition(self):
        """Test sentiment condition evaluation"""
        condition = SentimentCondition(threshold=-0.5, operator="lt")
        
        # Should match negative sentiment
        context = {'sentiment_polarity': -0.8}
        assert condition.evaluate(context) is True
        
        # Should not match positive sentiment
        context = {'sentiment_polarity': 0.3}
        assert condition.evaluate(context) is False
    
    def test_rating_condition(self):
        """Test rating condition evaluation"""
        condition = RatingCondition(threshold=3, operator="lt")
        
        # Should match low rating
        context = {'rating': 2}
        assert condition.evaluate(context) is True
        
        # Should not match high rating
        context = {'rating': 4}
        assert condition.evaluate(context) is False
    
    def test_keyword_condition(self):
        """Test keyword condition evaluation"""
        condition = KeywordCondition(keywords=['mauvais', 'défectueux'], match_any=True)
        
        # Should match if any keyword present
        context = {'text': 'le service était mauvais'}
        assert condition.evaluate(context) is True
        
        # Should not match if no keywords present
        context = {'text': 'le service était excellent'}
        assert condition.evaluate(context) is False
    
    def test_aspect_condition(self):
        """Test aspect condition evaluation"""
        condition = AspectCondition(aspects=['service'], match_any=True)
        
        # Should match if aspect present
        context = {'aspects': ['service', 'produit']}
        assert condition.evaluate(context) is True
        
        # Should not match if aspect not present
        context = {'aspects': ['livraison']}
        assert condition.evaluate(context) is False
    
    def test_add_remove_rule(self, rules_engine):
        """Test adding and removing rules"""
        rule = BusinessRule(
            name="test_rule",
            description="Test rule",
            conditions=[{"type": "rating", "threshold": 3, "operator": "lt"}],
            actions=[{"type": ActionType.CATEGORIZE, "category": "negative"}],
            priority=Priority.MEDIUM
        )
        
        # Add rule
        rules_engine.add_rule(rule)
        assert len(rules_engine.rules) == 1
        assert rules_engine.rules[0].name == "test_rule"
        
        # Remove rule
        rules_engine.remove_rule("test_rule")
        assert len(rules_engine.rules) == 0
    
    def test_evaluate_rule(self, rules_engine, sample_context):
        """Test individual rule evaluation"""
        rule = BusinessRule(
            name="negative_review",
            description="Detect negative reviews",
            conditions=[
                {"type": "rating", "threshold": 3, "operator": "lt"},
                {"type": "sentiment", "threshold": -0.5, "operator": "lt"}
            ],
            actions=[
                {"type": ActionType.ESCALATE, "department": "support"},
                {"type": ActionType.CATEGORIZE, "category": "negative"}
            ],
            priority=Priority.HIGH
        )
        
        result = rules_engine.evaluate_rule(rule, sample_context)
        
        assert result.matched is True
        assert result.rule_name == "negative_review"
        assert len(result.actions_triggered) == 2
    
    def test_process_context(self, rules_engine, sample_context):
        """Test processing context against multiple rules"""
        # Add multiple rules
        rule1 = BusinessRule(
            name="negative_sentiment",
            description="Negative sentiment rule",
            conditions=[{"type": "sentiment", "threshold": -0.5, "operator": "lt"}],
            actions=[{"type": ActionType.CATEGORIZE, "category": "negative"}],
            priority=Priority.MEDIUM
        )
        
        rule2 = BusinessRule(
            name="service_issue",
            description="Service issue rule",
            conditions=[{"type": "aspect", "aspects": ["service"], "match_any": True}],
            actions=[{"type": ActionType.ESCALATE, "department": "service"}],
            priority=Priority.HIGH
        )
        
        rules_engine.add_rule(rule1)
        rules_engine.add_rule(rule2)
        
        results = rules_engine.process(sample_context)
        
        # Both rules should match
        assert len(results) == 2
        matched_rules = {r.rule_name for r in results}
        assert "negative_sentiment" in matched_rules
        assert "service_issue" in matched_rules
    
    def test_execute_actions(self, rules_engine, sample_context):
        """Test action execution"""
        rule = BusinessRule(
            name="test_actions",
            description="Test action execution",
            conditions=[{"type": "rating", "threshold": 5, "operator": "lt"}],
            actions=[
                {"type": ActionType.CATEGORIZE, "category": "needs_attention"},
                {"type": ActionType.AUTO_RESPONSE, "template": "generic"},
                {"type": ActionType.FLAG_REVIEW, "flag": "manual_check"}
            ],
            priority=Priority.MEDIUM
        )
        
        rules_engine.add_rule(rule)
        results = rules_engine.process(sample_context)
        execution_summary = rules_engine.execute_actions(results)
        
        assert execution_summary['total_actions'] == 3
        assert execution_summary['successful_actions'] >= 0
        assert len(execution_summary['action_details']) == 3
    
    def test_rule_priority_ordering(self, rules_engine):
        """Test that rules are processed in priority order"""
        # Add rules with different priorities
        high_rule = BusinessRule(
            name="high_priority",
            description="High priority rule",
            conditions=[{"type": "rating", "threshold": 5, "operator": "lt"}],
            actions=[{"type": ActionType.CATEGORIZE, "category": "high"}],
            priority=Priority.HIGH
        )
        
        low_rule = BusinessRule(
            name="low_priority", 
            description="Low priority rule",
            conditions=[{"type": "rating", "threshold": 5, "operator": "lt"}],
            actions=[{"type": ActionType.CATEGORIZE, "category": "low"}],
            priority=Priority.LOW
        )
        
        # Add in reverse order
        rules_engine.add_rule(low_rule)
        rules_engine.add_rule(high_rule)
        
        context = {'rating': 2}
        results = rules_engine.process(context)
        
        # High priority rule should be processed first
        assert results[0].rule_name == "high_priority"
        assert results[1].rule_name == "low_priority"
    
    def test_inactive_rule(self, rules_engine):
        """Test that inactive rules are not processed"""
        rule = BusinessRule(
            name="inactive_rule",
            description="This rule is inactive",
            conditions=[{"type": "rating", "threshold": 5, "operator": "lt"}],
            actions=[{"type": ActionType.CATEGORIZE, "category": "test"}],
            priority=Priority.MEDIUM,
            active=False  # Rule is inactive
        )
        
        rules_engine.add_rule(rule)
        results = rules_engine.process({'rating': 2})
        
        # Should not match any rules
        assert len(results) == 0
    
    def test_get_rule_statistics(self, rules_engine):
        """Test rule statistics generation"""
        # Add rules with different priorities
        rules_engine.add_rule(BusinessRule(
            name="high1", description="", conditions=[], actions=[], priority=Priority.HIGH
        ))
        rules_engine.add_rule(BusinessRule(
            name="medium1", description="", conditions=[], actions=[], priority=Priority.MEDIUM
        ))
        rules_engine.add_rule(BusinessRule(
            name="low1", description="", conditions=[], actions=[], priority=Priority.LOW,
            active=False
        ))
        
        stats = rules_engine.get_rule_statistics()
        
        assert stats['total_rules'] == 3
        assert stats['active_rules'] == 2  # One is inactive
        assert stats['rules_by_priority']['HIGH'] == 1
        assert stats['rules_by_priority']['MEDIUM'] == 1
        assert stats['rules_by_priority']['LOW'] == 1
