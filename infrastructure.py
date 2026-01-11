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
    INFRASTRUCTURE = "Infrastructure Resilience"
    MAINTENANCE = "Infrastructure Maintenance"
    OPERATIONS = "Infrastructure Operations"
    MODERNIZATION = "Infrastructure Modernization"
    ASSESSMENT = "Infrastructure Assessment"
    PLANNING = "Infrastructure Planning"
    INTEGRATION = "Systems Integration"
    ADAPTATION = "Climate Adaptation"

@dataclass
class QuestionTemplate:
    relevant_professions: Set[str]
    category_type: CategoryType
    topic_category: TopicCategory
    template: str
    research_method: ResearchMethod

class InfrastructureHazardQuestionGenerator:
    def __init__(self):
        self.profession_groups = {
            "TRANSPORTATION": {
                "Highway Engineer",
                "Bridge Inspector",
                "Railway Systems Engineer",
                "Transit Operations Manager",
                "Airport Infrastructure Manager",
                "Port Facility Manager",
                "Transportation Safety Inspector",
                "Traffic Systems Engineer",
                "Pavement Engineer",
                "Transportation Planner"
            },
            "WATER": {
                "Water Systems Engineer",
                "Hydraulic Engineer",
                "Dam Safety Inspector",
                "Wastewater Treatment Specialist",
                "Maritime Infrastructure Manager",
                "Stormwater Engineer",
                "Water Quality Specialist",
                "Coastal Infrastructure Engineer"
            },
            "ENERGY": {
                "Power Systems Engineer",
                "Electrical Grid Manager",
                "Energy Distribution Specialist",
                "EV Infrastructure Planner",
                "Renewable Energy Systems Manager",
                "Transmission Line Engineer",
                "Substation Engineer",
                "Energy Storage Specialist"
            },
            "BUILDINGS": {
                "Structural Engineer",
                "Building Systems Manager",
                "Facilities Manager",
                "Real Estate Asset Manager",
                "Building Automation Specialist",
                "Construction Manager",
                "Building Code Inspector",
                "MEP Systems Engineer"
            },
            "COMMUNICATIONS": {
                "Telecommunications Engineer",
                "Broadband Infrastructure Specialist",
                "Network Resilience Manager",
                "Data Center Infrastructure Engineer",
                "Fiber Optics Specialist",
                "Communications Systems Planner",
                "Network Security Engineer"
            }
        }

        self.hazard_types = [
                                "cold wave",
                                "heat wave",
                                "coastal flooding",
                                "ice storm",
                                "hurricane",
                                "drought",
                                "wildfire"
                            ]
        
        
        self.hazard_location_mapping = {
            "coastal flooding": [
            "Bergen, New Jersey",
            "Atlantic, New Jersey",
            "Ocean, New Jersey",
            "Cape May, New Jersey",
            "Hudson, New Jersey",
            "Monmouth, New Jersey",
            "Grays Harbor, Washington",
            "Middlesex, New Jersey",
            "Kings, New York",
            "Cumberland, New Jersey",
            "Clatsop, Oregon",
            "Cameron, Texas",
            "Philadelphia, Pennsylvania",
            "Queens, New York",
            "Coos, Oregon",
            "Bronx, New York",
            "Sussex, Delaware",
            "Westchester, New York",
            "New York, New York",
            "Jefferson, Louisiana",
            "Fairfield, Connecticut",
            "St. Charles, Louisiana",
            "Suffolk, New York",
            "Aransas, Texas",
            "Union, New Jersey",
        ],
        "cold wave": [
            "Cook, Illinois",
            "Milwaukee, Wisconsin",
            "Minnehaha, South Dakota",
            "Wayne, Michigan",
            "Lake, Illinois",
            "Nueces, Texas",
            "Lake, Indiana",
            "Hennepin, Minnesota",
            "Williams, North Dakota",
            "Will, Illinois",
            "Yakima, Washington",
            "Anoka, Minnesota",
            "Flathead, Montana",
            "Winnebago, Illinois",
            "Pennington, South Dakota",
            "Dane, Wisconsin",
            "Ramsey, Minnesota",
            "Cass, North Dakota",
            "Marathon, Wisconsin",
            "Sheboygan, Wisconsin",
            "Blue Earth, Minnesota",
            "Brown, Wisconsin",
            "Olmsted, Minnesota",
            "Outagamie, Wisconsin",
            "St. Louis, Minnesota",
        ],
        "drought": [
            "Santa Barbara, California",
            "Yolo, California",
            "Sutter, California",
            "Napa, California",
            "Colusa, California",
            "Glenn, California",
            "Butte, California",
            "Sonoma, California",
            "Sacramento, California",
            "Solano, California",
            "Pinal, Arizona",
            "Floyd, Texas",
            "Lubbock, Texas",
            "Humboldt, Nevada",
            "Doña Ana, New Mexico",
            "Maricopa, Arizona",
            "Yuma, Arizona",
            "Kings, California",
            "Imperial, California",
            "Merced, California",
            "Madera, California",
            "Stanislaus, California",
            "Fresno, California",
            "Tulare, California",
            "Kern, California",
        ],
        "heat wave": [
            "Cook, Illinois",
            "Clark, Nevada",
            "St. Louis, Missouri",
            "Philadelphia, Pennsylvania",
            "Dallas, Texas",
            "Tulsa, Oklahoma",
            "Maricopa, Arizona",
            "Queens, New York",
            "Tarrant, Texas",
            "Kings, New York",
            "Oklahoma, Oklahoma",
            "Tulare, California",
            "Jackson, Missouri",
            "Shelby, Tennessee",
            "Baltimore, Maryland",
            "Fulton, Georgia",
            "Los Angeles, California",
            "Harris, Texas",
            "Bexar, Texas",
            "Fairfax, Virginia",
            "Franklin, Ohio",
            "DeKalb, Georgia",
            "Prince George's, Maryland",
            "Mecklenburg, North Carolina",
            "Wayne, Michigan",
        ],
        "hurricane": [
            "Harris, Texas",
            "Miami-Dade, Florida",
            "Broward, Florida",
            "Palm Beach, Florida",
            "Hillsborough, Florida",
            "Lee, Florida",
            "Brevard, Florida",
            "Pinellas, Florida",
            "Charleston, South Carolina",
            "Pasco, Florida",
            "Horry, South Carolina",
            "Collier, Florida",
            "Chatham, Georgia",
            "Mobile, Alabama",
            "New Hanover, North Carolina",
            "Galveston, Texas",
            "Orange, Florida",
            "Volusia, Florida",
            "Indian River, Florida",
            "St. Lucie, Florida",
            "St. Johns, Florida",
            "Manatee, Florida",
            "Clay, Florida",
            "Beaufort, South Carolina",
            "Escambia, Florida",
        ],
        "ice storm": [
            "Nassau, New York",
            "Tulsa, Oklahoma",
            "Greene, Missouri",
            "Lancaster, Nebraska",
            "St. Louis, Missouri",
            "Oakland, Michigan",
            "Boone, Missouri",
            "Richland, South Carolina",
            "Monmouth, New Jersey",
            "Washington, Arkansas",
            "Macomb, Michigan",
            "Johnson, Kansas",
            "Morris, Texas",
            "Baxter, Arkansas",
            "Rogers, Oklahoma",
            "Douglas, Nebraska",
            "Sedgwick, Kansas",
            "Linn, Iowa",
            "Dubuque, Iowa",
            "Stark, Ohio",
            "Polk, Iowa",
            "Peoria, Illinois",
            "Knox, Tennessee",
            "Lucas, Ohio",
            "Hamilton, Ohio",
        ],
        "wildfire": [
            "San Diego, California",
            "Riverside, California",
            "San Bernardino, California",
            "Los Angeles, California",
            "Washington, Utah",
            "Elko, Nevada",
            "Ventura, California",
            "Orange, California",
            "Pima, Arizona",
            "Maricopa, Arizona",
            "Ravalli, Montana",
            "Kern, California",
            "Yavapai, Arizona",
            "Utah, Utah",
            "Madera, California",
            "Nevada, California",
            "Placer, California",
            "Shasta, California",
            "Siskiyou, California",
            "Tehama, California",
            "Santa Cruz, California",
            "Alameda, California",
            "Tuolumne, California",
             ],
         }
    

        
        self.time_periods = ["5 - 10 years", "10 - 20 years", "20 - 30 years", "30 - 50 years", "50 - 100 years"]
        
        self.infrastructure_types = {
            "TRANSPORTATION": [
                "highway network", "bridge system", "public transit system",
                "railway infrastructure", "airport facilities", "port facilities",
                "freight terminals", "traffic control systems"
            ],
            "WATER": [
                "water treatment plant", "wastewater system", "dam infrastructure",
                "stormwater system", "coastal protection", "water distribution network"
            ],
            "ENERGY": [
                "electrical grid", "power distribution network", "EV charging network",
                "renewable energy infrastructure", "energy storage facilities",
                "power transmission lines", "substations"
            ],
            "BUILDINGS": [
                "public buildings", "critical facilities", "commercial structures",
                "government facilities", "healthcare infrastructure", 
                "educational facilities"
            ],
            "COMMUNICATIONS": [
                "broadband network", "fiber optic infrastructure", "data centers",
                "cellular networks", "emergency communications systems",
                "telecommunications facilities"
            ]
        }
        
        self.assessment_metrics = {
            "PHYSICAL": [
                "structural integrity", "material durability", "physical condition",
                "deterioration rate", "structural capacity"
            ],
            "OPERATIONAL": [
                "operational efficiency", "system reliability", "service continuity",
                "performance capacity", "response time"
            ],
            "MAINTENANCE": [
                "maintenance requirements", "repair needs", "replacement schedule",
                "lifecycle costs", "preventive maintenance"
            ],
            "RESILIENCE": [
                "hazard resistance", "recovery capability", "adaptation capacity",
                "redundancy level", "vulnerability index"
            ]
        }
        
        self.templates = self._create_templates()

    def _create_templates(self) -> List[QuestionTemplate]:
        templates = []
        
        # Transportation Infrastructure Templates
        templates.extend([
            QuestionTemplate(
                relevant_professions=self.profession_groups["TRANSPORTATION"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.ASSESSMENT,
                template="How will [HAZARD] impact the [INFRASTRUCTURE_TYPE]'s [ASSESSMENT_METRIC] in [LOCATION]?",
                research_method=ResearchMethod.BOTH
            ),
            QuestionTemplate(
                relevant_professions=self.profession_groups["TRANSPORTATION"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.MODERNIZATION,
                template="What modernization measures are needed for [INFRASTRUCTURE_TYPE] to withstand [HAZARD] conditions in [LOCATION]?",
                research_method=ResearchMethod.LITERATURE_REVIEW
            ),
        ])
        
        # Water Infrastructure Templates
        templates.extend([
            QuestionTemplate(
                relevant_professions=self.profession_groups["WATER"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.INFRASTRUCTURE,
                template="How vulnerable is the [INFRASTRUCTURE_TYPE] to [HAZARD] in [LOCATION]?",
                research_method=ResearchMethod.DATA_ANALYSIS
            ),
            QuestionTemplate(
                relevant_professions=self.profession_groups["WATER"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.MAINTENANCE,
                template="What maintenance strategies should be implemented for [INFRASTRUCTURE_TYPE] to mitigate [HAZARD] impacts in [LOCATION]?",
                research_method=ResearchMethod.BOTH
            ),
        ])
        
        # Energy Infrastructure Templates
        templates.extend([
            QuestionTemplate(
                relevant_professions=self.profession_groups["ENERGY"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.OPERATIONS,
                template="How will [HAZARD] affect [INFRASTRUCTURE_TYPE] reliability in [LOCATION]?",
                research_method=ResearchMethod.DATA_ANALYSIS
            ),
            QuestionTemplate(
                relevant_professions=self.profession_groups["ENERGY"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.ADAPTATION,
                template="What adaptation strategies should be implemented for [INFRASTRUCTURE_TYPE] to handle [HAZARD] in [LOCATION]?",
                research_method=ResearchMethod.LITERATURE_REVIEW
            ),
        ])
        
        # Buildings Infrastructure Templates
        templates.extend([
            QuestionTemplate(
                relevant_professions=self.profession_groups["BUILDINGS"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.ASSESSMENT,
                template="What is the projected impact of [HAZARD] on [INFRASTRUCTURE_TYPE] in [LOCATION] over [TIME_PERIOD]?",
                research_method=ResearchMethod.BOTH
            ),
            QuestionTemplate(
                relevant_professions=self.profession_groups["BUILDINGS"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.PLANNING,
                template="How should [INFRASTRUCTURE_TYPE] design standards evolve to address [HAZARD] risks in [LOCATION]?",
                research_method=ResearchMethod.LITERATURE_REVIEW
            ),
        ])
        
        # Communications Infrastructure Templates
        templates.extend([
            QuestionTemplate(
                relevant_professions=self.profession_groups["COMMUNICATIONS"],
                category_type=CategoryType.RECOMMENDATION,
                topic_category=TopicCategory.MODERNIZATION,
                template="How should [INFRASTRUCTURE_TYPE] be upgraded to maintain service during [HAZARD] events in [LOCATION]?",
                research_method=ResearchMethod.BOTH
            ),
            QuestionTemplate(
                relevant_professions=self.profession_groups["COMMUNICATIONS"],
                category_type=CategoryType.FACT_BASED,
                topic_category=TopicCategory.INFRASTRUCTURE,
                template="What are the critical vulnerabilities of [INFRASTRUCTURE_TYPE] to [HAZARD] in [LOCATION]?",
                research_method=ResearchMethod.DATA_ANALYSIS
            ),
        ])
        
        # Cross-sector Templates
        templates.extend([
            QuestionTemplate(
                relevant_professions=self.profession_groups["TRANSPORTATION"] | self.profession_groups["ENERGY"],
                category_type=CategoryType.HYBRID,
                topic_category=TopicCategory.INTEGRATION,
                template="How will [HAZARD] affect the interdependencies between [INFRASTRUCTURE_TYPE] and energy systems in [LOCATION]?",
                research_method=ResearchMethod.BOTH
            ),
            QuestionTemplate(
                relevant_professions=self.profession_groups["WATER"] | self.profession_groups["BUILDINGS"],
                category_type=CategoryType.HYBRID,
                topic_category=TopicCategory.INTEGRATION,
                template="What are the cascading impacts of [HAZARD] between [INFRASTRUCTURE_TYPE] and building systems in [LOCATION]?",
                research_method=ResearchMethod.BOTH
            ),
        ])
        
        return templates

    def generate_question(
        self,
        template_index: int = None,
        hazard_type: str = None,
        research_method: ResearchMethod = None,
        topic_category: TopicCategory = None,
        category_type: CategoryType = None,
        profession_group: str = None,
        specific_profession: str = None,
        infrastructure_type: str = None,
        location: str = None,
        assessment_metric_category: str = None
    ) -> Dict[str, str]:
        """
        Generate a question with specific filtering criteria.
        
        Args:
            template_index (int, optional): Specific template index to use
            hazard_type (str, optional): Specific hazard type to include
            research_method (ResearchMethod, optional): Filter by research method
            topic_category (TopicCategory, optional): Filter by topic category
            category_type (CategoryType, optional): Filter by category type
            profession_group (str, optional): Filter by profession group (e.g., "TRANSPORTATION")
            specific_profession (str, optional): Filter by specific profession
            infrastructure_type (str, optional): Filter by specific infrastructure type
            location (str, optional): Filter by specific location
            assessment_metric_category (str, optional): Filter by assessment metric category
            
        Returns:
            Dict[str, str]: Generated question with metadata
        """
        # Filter templates based on criteria
        available_templates = self.templates.copy()
        
        if research_method:
            available_templates = [t for t in available_templates if t.research_method == research_method]
        
        if topic_category:
            available_templates = [t for t in available_templates if t.topic_category == topic_category]
        
        if category_type:
            available_templates = [t for t in available_templates if t.category_type == category_type]
        
        if profession_group:
            available_templates = [
                t for t in available_templates 
                if any(prof in self.profession_groups[profession_group] for prof in t.relevant_professions)
            ]
        
        if specific_profession:
            available_templates = [
                t for t in available_templates if specific_profession in t.relevant_professions
            ]
        
        if not available_templates:
            raise ValueError("No templates match the specified criteria")

        # Select template
        if template_index is not None and template_index < len(available_templates):
            template = available_templates[template_index]
        else:
            template = random.choice(available_templates)

        # Prepare variables for template filling
        variables = {}
        
        # Handle hazard type
        if hazard_type and hazard_type in self.hazard_types:
            variables["[HAZARD]"] = hazard_type
        else:
            variables["[HAZARD]"] = random.choice(self.hazard_types)
        
        # Handle location
        if location and location in self.hazard_location_mapping[variables["[HAZARD]"]]:
            variables["[LOCATION]"] = location
        else:
            variables["[LOCATION]"] = random.choice(self.hazard_location_mapping[variables["[HAZARD]"]])
        
        # Handle time period
        variables["[TIME_PERIOD]"] = random.choice(self.time_periods)
        
        # Handle infrastructure type
        if infrastructure_type:
            for group_types in self.infrastructure_types.values():
                if infrastructure_type in group_types:
                    variables["[INFRASTRUCTURE_TYPE]"] = infrastructure_type
                    break
            if "[INFRASTRUCTURE_TYPE]" not in variables:
                raise ValueError("Specified infrastructure type not found")
        else:
            # Determine relevant infrastructure types based on the profession group
            relevant_group = None
            for group, professions in self.profession_groups.items():
                if any(prof in template.relevant_professions for prof in professions):
                    relevant_group = group
                    break
            
            variables["[INFRASTRUCTURE_TYPE]"] = random.choice(
                self.infrastructure_types[relevant_group] if relevant_group else 
                [item for sublist in self.infrastructure_types.values() for item in sublist]
            )
        
        # Handle assessment metric
        if assessment_metric_category and assessment_metric_category in self.assessment_metrics:
            variables["[ASSESSMENT_METRIC]"] = random.choice(
                self.assessment_metrics[assessment_metric_category]
            )
        else:
            metric_category = random.choice(list(self.assessment_metrics.keys()))
            variables["[ASSESSMENT_METRIC]"] = random.choice(
                self.assessment_metrics[metric_category]
            )

        # Fill template
        filled_template = template.template
        for key, value in variables.items():
            if key in filled_template:
                filled_template = filled_template.replace(key, value)

        # Select profession
        if specific_profession and specific_profession in template.relevant_professions:
            chosen_profession = specific_profession
        else:
            chosen_profession = random.choice(list(template.relevant_professions))

        return {
            "profession": chosen_profession,
            "category_type": template.category_type.value,
            "topic_category": template.topic_category.value,
            'location': variables["[LOCATION]"],
            'timeline': variables["[TIME_PERIOD]"],
            'hazard_type': variables["[HAZARD]"],
            'infrastructure_type': variables["[INFRASTRUCTURE_TYPE]"],
            "question": filled_template,
            "research_method": template.research_method.value,
            "relevant_professions": list(template.relevant_professions)
        }

    def generate_questions(
        self,
        n: int = 5,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate multiple questions with the same filtering criteria.
        
        Args:
            n (int): Number of questions to generate
            **kwargs: Any filtering criteria supported by generate_question
            
        Returns:
            List[Dict[str, str]]: List of generated questions with metadata
        """
        return [self.generate_question(**kwargs) for _ in range(n)]

def main():
    generator = InfrastructureHazardQuestionGenerator()
    
    print("Infrastructure Professional Groups and Their Specialists:")
    for group, professions in generator.profession_groups.items():
        print(f"\n{group}:")
        for prof in sorted(professions):
            print(f"  - {prof}")

    print("\nGenerating Infrastructure-Focused Questions:")
    questions = generator.generate_questions(n=5)
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"Selected Profession: {q['profession']}")
        print(f"Category Type: {q['category_type']}")
        print(f"Topic Category: {q['topic_category']}")
        print(f"Location: {q['location']}")
        print(f"Timeline: {q['timeline']}")
        print(f"Hazard Type: {q['hazard_type']}")
        print(f"Infrastructure Type: {q['infrastructure_type']}")
        print(f"Question: {q['question']}")
        print(f"Research Method: {q['research_method']}")
        print("Relevant Professions:")
        for prof in q['relevant_professions']:
            print(f"  - {prof}")

        # Notice that due to the random selection, the relevant professions (or even the selected profession) may not always make the most sense for the question, given the location, hazard type, and infrastructure type.
        

if __name__ == "__main__":
    main()