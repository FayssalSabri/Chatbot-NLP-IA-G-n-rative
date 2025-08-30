from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    AUTO_RESPONSE = "auto_response"
    ESCALATE = "escalate"
    FLAG_REVIEW = "flag_review"
    CATEGORIZE = "categorize"
    SEND_EMAIL = "send_email"
    UPDATE_STATUS = "update_status"

@dataclass
class BusinessRule:
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    priority: Priority = Priority.MEDIUM
    active: bool = True

@dataclass
class RuleResult:
    rule_name: str
    matched: bool
    actions_triggered: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

class Condition(ABC):
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> bool:
        pass

class SentimentCondition(Condition):
    def __init__(self, threshold: float, operator: str = "lt"):
        self.threshold = threshold
        self.operator = operator
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        sentiment = context.get('sentiment_polarity', 0)
        if self.operator == "lt":
            return sentiment < self.threshold
        elif self.operator == "gt":
            return sentiment > self.threshold
        elif self.operator == "eq":
            return abs(sentiment - self.threshold) < 0.1
        return False

class RatingCondition(Condition):
    def __init__(self, threshold: int, operator: str = "lt"):
        self.threshold = threshold
        self.operator = operator
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        rating = context.get('rating', 0)
        if self.operator == "lt":
            return rating < self.threshold
        elif self.operator == "gt":
            return rating > self.threshold
        elif self.operator == "eq":
            return rating == self.threshold
        return False

class KeywordCondition(Condition):
    def __init__(self, keywords: List[str], match_any: bool = True):
        self.keywords = [kw.lower() for kw in keywords]
        self.match_any = match_any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        text = context.get('text', '').lower()
        if self.match_any:
            return any(keyword in text for keyword in self.keywords)
        else:
            return all(keyword in text for keyword in self.keywords)

class AspectCondition(Condition):
    def __init__(self, aspects: List[str], match_any: bool = True):
        self.aspects = aspects
        self.match_any = match_any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        detected_aspects = context.get('aspects', [])
        if self.match_any:
            return any(aspect in detected_aspects for aspect in self.aspects)
        else:
            return all(aspect in detected_aspects for aspect in self.aspects)

class BusinessRulesEngine:
    def __init__(self):
        self.rules: List[BusinessRule] = []
        self.condition_registry = {
            'sentiment': SentimentCondition,
            'rating': RatingCondition,
            'keyword': KeywordCondition,
            'aspect': AspectCondition
        }
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default business rules"""
        # Rule 1: Critical negative reviews
        self.add_rule(BusinessRule(
            name="critical_negative_review",
            description="Flag critical negative reviews for immediate attention",
            conditions=[
                {"type": "rating", "threshold": 3, "operator": "lt"},
                {"type": "sentiment", "threshold": -0.5, "operator": "lt"}
            ],
            actions=[
                {"type": ActionType.ESCALATE, "department": "customer_service", "priority": "high"},
                {"type": ActionType.FLAG_REVIEW, "flag": "critical_negative"}
            ],
            priority=Priority.CRITICAL
        ))
        
        # Rule 2: Service complaints
        self.add_rule(BusinessRule(
            name="service_complaint",
            description="Handle service-related complaints",
            conditions=[
                {"type": "aspect", "aspects": ["service"], "match_any": True},
                {"type": "keyword", "keywords": ["problème", "mauvais service", "incompétent"], "match_any": True}
            ],
            actions=[
                {"type": ActionType.CATEGORIZE, "category": "service_issue"},
                {"type": ActionType.AUTO_RESPONSE, "template": "service_apology"}
            ],
            priority=Priority.HIGH
        ))
        
        # Rule 3: Product quality issues
        self.add_rule(BusinessRule(
            name="quality_issue",
            description="Identify product quality problems",
            conditions=[
                {"type": "aspect", "aspects": ["qualité", "produit"], "match_any": True},
                {"type": "sentiment", "threshold": -0.3, "operator": "lt"}
            ],
            actions=[
                {"type": ActionType.CATEGORIZE, "category": "quality_issue"},
                {"type": ActionType.FLAG_REVIEW, "flag": "quality_check_needed"}
            ],
            priority=Priority.MEDIUM
        ))
        
        # Rule 4: Positive feedback
        self.add_rule(BusinessRule(
            name="positive_feedback",
            description="Handle positive feedback",
            conditions=[
                {"type": "rating", "threshold": 4, "operator": "gt"},
                {"type": "sentiment", "threshold": 0.3, "operator": "gt"}
            ],
            actions=[
                {"type": ActionType.CATEGORIZE, "category": "positive_feedback"},
                {"type": ActionType.AUTO_RESPONSE, "template": "thank_you"}
            ],
            priority=Priority.LOW
        ))
        
        # Rule 5: Delivery complaints
        self.add_rule(BusinessRule(
            name="delivery_complaint",
            description="Handle delivery-related issues",
            conditions=[
                {"type": "aspect", "aspects": ["livraison"], "match_any": True},
                {"type": "keyword", "keywords": ["retard", "pas reçu", "délai"], "match_any": True}
            ],
            actions=[
                {"type": ActionType.CATEGORIZE, "category": "delivery_issue"},
                {"type": ActionType.ESCALATE, "department": "logistics", "priority": "medium"}
            ],
            priority=Priority.MEDIUM
        ))
    
    def add_rule(self, rule: BusinessRule):
        """Add a business rule"""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a business rule"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed rule: {rule_name}")
    
    def _create_condition(self, condition_config: Dict[str, Any]) -> Condition:
        """Create condition instance from configuration"""
        condition_type = condition_config['type']
        
        if condition_type not in self.condition_registry:
            raise ValueError(f"Unknown condition type: {condition_type}")
        
        condition_class = self.condition_registry[condition_type]
        config = {k: v for k, v in condition_config.items() if k != 'type'}
        
        return condition_class(**config)
    
    def evaluate_rule(self, rule: BusinessRule, context: Dict[str, Any]) -> RuleResult:
        """Evaluate a single rule against context"""
        if not rule.active:
            return RuleResult(rule.name, False, [])
        
        try:
            # Evaluate all conditions
            all_conditions_met = True
            
            for condition_config in rule.conditions:
                condition = self._create_condition(condition_config)
                if not condition.evaluate(context):
                    all_conditions_met = False
                    break
            
            if all_conditions_met:
                return RuleResult(
                    rule_name=rule.name,
                    matched=True,
                    actions_triggered=rule.actions,
                    metadata={"priority": rule.priority.name}
                )
            else:
                return RuleResult(rule.name, False, [])
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return RuleResult(rule.name, False, [])
    
    def process(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Process context against all rules"""
        results = []
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority.value, reverse=True)
        
        for rule in sorted_rules:
            result = self.evaluate_rule(rule, context)
            if result.matched:
                results.append(result)
                logger.info(f"Rule triggered: {rule.name}")
        
        return results
    
    def execute_actions(self, results: List[RuleResult]) -> Dict[str, Any]:
        """Execute actions from rule results"""
        execution_summary = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'action_details': []
        }
        
        for result in results:
            for action in result.actions_triggered:
                try:
                    action_result = self._execute_action(action, result)
                    execution_summary['successful_actions'] += 1
                    execution_summary['action_details'].append({
                        'rule': result.rule_name,
                        'action': action,
                        'status': 'success',
                        'result': action_result
                    })
                except Exception as e:
                    logger.error(f"Failed to execute action {action}: {e}")
                    execution_summary['failed_actions'] += 1
                    execution_summary['action_details'].append({
                        'rule': result.rule_name,
                        'action': action,
                        'status': 'failed',
                        'error': str(e)
                    })
                
                execution_summary['total_actions'] += 1
        
        return execution_summary
    
    def _execute_action(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Execute a single action"""
        action_type = action.get('type')
        
        if action_type == ActionType.ESCALATE:
            return self._escalate_issue(action, rule_result)
        elif action_type == ActionType.AUTO_RESPONSE:
            return self._generate_auto_response(action, rule_result)
        elif action_type == ActionType.CATEGORIZE:
            return self._categorize_item(action, rule_result)
        elif action_type == ActionType.FLAG_REVIEW:
            return self._flag_for_review(action, rule_result)
        elif action_type == ActionType.SEND_EMAIL:
            return self._send_email_notification(action, rule_result)
        elif action_type == ActionType.UPDATE_STATUS:
            return self._update_status(action, rule_result)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _escalate_issue(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Escalate issue to appropriate department"""
        department = action.get('department', 'general')
        priority = action.get('priority', 'medium')
        
        # In a real implementation, this would integrate with ticketing system
        logger.info(f"Escalating to {department} with priority {priority}")
        
        return {
            'action': 'escalated',
            'department': department,
            'priority': priority,
            'timestamp': 'now'  # Would use actual timestamp
        }
    
    def _generate_auto_response(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Generate automatic response"""
        template = action.get('template', 'generic')
        
        templates = {
            'service_apology': "Nous sommes désolés pour les inconvénients rencontrés avec notre service. Nous allons examiner votre cas rapidement.",
            'thank_you': "Merci pour votre retour positif ! Nous sommes ravis que vous soyez satisfait.",
            'generic': "Nous avons bien reçu votre message et nous vous répondrons rapidement."
        }
        
        response = templates.get(template, templates['generic'])
        
        return {
            'action': 'auto_response_generated',
            'template': template,
            'response': response
        }
    
    def _categorize_item(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Categorize the item"""
        category = action.get('category', 'general')
        
        return {
            'action': 'categorized',
            'category': category
        }
    
    def _flag_for_review(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Flag item for manual review"""
        flag = action.get('flag', 'review_needed')
        
        return {
            'action': 'flagged',
            'flag': flag
        }
    
    def _send_email_notification(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Send email notification"""
        recipient = action.get('recipient', 'admin@company.com')
        subject = action.get('subject', 'Rule Triggered Notification')
        
        # In real implementation, integrate with email service
        logger.info(f"Sending email to {recipient}: {subject}")
        
        return {
            'action': 'email_sent',
            'recipient': recipient,
            'subject': subject
        }
    
    def _update_status(self, action: Dict[str, Any], rule_result: RuleResult) -> Dict[str, Any]:
        """Update item status"""
        new_status = action.get('status', 'processed')
        
        return {
            'action': 'status_updated',
            'new_status': new_status
        }
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rules"""
        return {
            'total_rules': len(self.rules),
            'active_rules': len([r for r in self.rules if r.active]),
            'rules_by_priority': {
                priority.name: len([r for r in self.rules if r.priority == priority])
                for priority in Priority
            }
        }

# Example usage
if __name__ == "__main__":
    # Sample context
    sample_context = {
        'text': 'Le service client était vraiment mauvais, très déçu de mon achat',
        'rating': 2,
        'sentiment_polarity': -0.7,
        'aspects': ['service'],
        'entities': []
    }
    
    # Create rules engine
    rules_engine = BusinessRulesEngine()
    
    # Process context
    results = rules_engine.process(sample_context)
    
    print(f"Matched {len(results)} rules:")
    for result in results:
        print(f"- {result.rule_name}: {len(result.actions_triggered)} actions")
    
    # Execute actions
    execution_summary = rules_engine.execute_actions(results)
    print(f"\nExecution summary: {execution_summary['successful_actions']}/{execution_summary['total_actions']} actions successful")

