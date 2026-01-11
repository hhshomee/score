from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum
import random

class CategoryType(Enum):
    FACT_BASED = "Fact-Based"
    RECOMMENDATION = "Recommendation-Seeking"
    HYBRID = "Hybrid"

class ResearchMethod(Enum):
    DATA_ANALYSIS = "Data Analysis"
    LITERATURE_REVIEW = "Literature Review"
    BOTH = "Both"

class TopicCategory(Enum):
    CLIMATE_DATA = "Climate and Environmental Data"
    HISTORICAL = "Historical and Current Conditions"
    DEMOGRAPHICS = "Demographics and Infrastructure"
    ASSESSMENT = "Assessment and Analysis"
    STRATEGY = "Strategy and Management"
    POLICY = "Policy and Planning"
    QUANTITATIVE = "Quantitative Measures"
    PARTNERSHIP = "Partnership and Resources"
    HYBRID = "Hybrid Questions"

@dataclass
class QuestionTemplate:
    relevant_professions: Set[str]  # Set of specific professions for this template
    category_type: CategoryType
    topic_category: TopicCategory
    template: str
    research_method: ResearchMethod

class HazardQuestionGenerator:
    def __init__(self):
        # Initialize profession groups based on original data
        self.profession_groups = {
            "SCIENTISTS": {
                "Atmospheric Scientist"
            },
            "PLANNERS": {
                "Urban Planner",
                "Hazard Mitigation Planner",
                "Urban Risk Manager"
            },
            "MANAGERS": {
                "Landscape Manager",
                "Public Safety Manager"
            },
            "ANALYSTS": {
                "Emergency Management Data Analyst",
                "Climate Change Risk Analyst"
            },
            "HOMEOWNERS": {
                "Homeowner"
            }
        }

        # Initialize variable pools
        self.hazard_types = ["wildfire", "drought", "heat wave", "extreme precipitation", "flooding"]
        self.locations = ["Boston", "Denver", "Chicago", "Los Angeles", "Miami"]
        self.time_periods = ["50 years", "100 years", "150 years"]
        self.development_types = ["urban", "rural", "infrastructure", "housing"]
        self.policy_types = ["zoning codes", "building codes", "emergency protocols"]
        self.resource_types = ["water resources", "infrastructure", "public health"]
        
        # Initialize templates
        self.templates = self._create_templates()

    def _create_templates(self) -> List[QuestionTemplate]:
        """Create the base question templates with specific profession associations"""
        return [
            # Climate Data Templates (Scientists focused)
            QuestionTemplate(
                relevant_professions=self.profession_groups["SCIENTISTS"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.CLIMATE_DATA,
                template="What's the climate's impact on [HAZARD] occurrences and intensity in [LOCATION]?",
                research_method=ResearchMethod.DATA_ANALYSIS
            ),
            # Historical Analysis Templates (Analysts focused)
            QuestionTemplate(
                relevant_professions=self.profession_groups["ANALYSTS"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.HISTORICAL,
                template="Has [HAZARD] frequency changed in [LOCATION] over the past [TIME_PERIOD]?",
                research_method=ResearchMethod.DATA_ANALYSIS
            ),
            # Demographics Templates (Planners focused)
            QuestionTemplate(
                relevant_professions=self.profession_groups["PLANNERS"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.DEMOGRAPHICS,
                template="Which areas in [LOCATION] are most vulnerable to [HAZARD] risks?",
                research_method=ResearchMethod.BOTH
            ),
            # Strategy Templates (Managers focused)
            QuestionTemplate(
                relevant_professions=self.profession_groups["MANAGERS"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.STRATEGY,
                template="What [HAZARD] management strategies should be implemented in [LOCATION]?",
                research_method=ResearchMethod.LITERATURE_REVIEW
            ),
            # Homeowner Templates
            QuestionTemplate(
                relevant_professions=self.profession_groups["HOMEOWNERS"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.STRATEGY,
                template="How should properties be managed to reduce [HAZARD] risk in [LOCATION]?",
                research_method=ResearchMethod.LITERATURE_REVIEW
            ),
            # Multi-stakeholder Templates
            QuestionTemplate(
                relevant_professions=self.profession_groups["PLANNERS"] | self.profession_groups["ANALYSTS"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.ASSESSMENT,
                template="What are the projected [HAZARD] risks for [LOCATION] under different climate scenarios?",
                research_method=ResearchMethod.DATA_ANALYSIS
            ),
            # Policy Templates
            QuestionTemplate(
                relevant_professions=self.profession_groups["PLANNERS"] | self.profession_groups["MANAGERS"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.POLICY,
                template="What changes to [POLICY_TYPE] are needed to address [HAZARD] risks in [LOCATION]?",
                research_method=ResearchMethod.LITERATURE_REVIEW
            )
        ]

    def _fill_template(self, template: QuestionTemplate) -> Dict[str, str]:
        """Fill in a template with specific variables"""
        variables = {
            "[HAZARD]": random.choice(self.hazard_types),
            "[LOCATION]": random.choice(self.locations),
            "[TIME_PERIOD]": random.choice(self.time_periods),
            "[DEVELOPMENT_TYPE]": random.choice(self.development_types),
            "[POLICY_TYPE]": random.choice(self.policy_types),
            "[RESOURCE_TYPE]": random.choice(self.resource_types)
        }

        filled_template = template.template
        # Replace variables in the template
        for key, value in variables.items():
            filled_template = filled_template.replace(key, value)

        # Choose a random relevant profession
        chosen_profession = random.choice(list(template.relevant_professions))

        return {
            "profession": chosen_profession,
            "category_type": template.category_type.value,
            "topic_category": template.topic_category.value,
            "question": filled_template,
            "research_method": template.research_method.value,
            "possible_professions": list(template.relevant_professions)
        }

    def generate_question(self, template_index: int = None, hazard_type: str = None) -> Dict[str, str]:
        """Generate a single question from either a specific template or a random one"""
        if template_index is None:
            template = random.choice(self.templates)
        else:
            template = self.templates[template_index]
        
        result = self._fill_template(template)
        if hazard_type:
            result["question"] = result["question"].replace(
                random.choice(self.hazard_types), hazard_type
            )
        return result

    def get_all_professions(self) -> Set[str]:
        """Get a set of all unique professions across all groups"""
        all_professions = set()
        for group in self.profession_groups.values():
            all_professions.update(group)
        return all_professions

    def generate_questions(self, n: int = 5, hazard_type: str = None) -> List[Dict[str, str]]:
        """Generate multiple questions, optionally for a specific hazard type"""
        questions = []
        for _ in range(n):
            questions.append(self.generate_question(hazard_type=hazard_type))
        return questions

def main():
    generator = HazardQuestionGenerator()
    
    # Print all available professions
    print("Available Professions:")
    for group, professions in generator.profession_groups.items():
        print(f"\n{group}:")
        for prof in professions:
            print(f"  - {prof}")

    print("\nGenerating random questions with relevant professions:")
    questions = generator.generate_questions(n=10)
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"Selected Profession: {q['profession']}")
        print(f"Category Type: {q['category_type']}")
        print(f"Topic Category: {q['topic_category']}")
        print(f"Question: {q['question']}")
        print(f"Research Method: {q['research_method']}")
        print("Relevant Professions:")
        for prof in q['possible_professions']:
            print(f"  - {prof}")

if __name__ == "__main__":
    main()