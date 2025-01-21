from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List, Optional


from dotenv import load_dotenv
import os


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')

# llm = LLM(
#     model='gpt-4o',
#     api_key=openai_api_key,
#     api_base='https://api.openai.com/v1'
# )


class Flight(BaseModel):
    name: str = Field(..., description="Name of the airline")
    number: str = Field(..., description="Flight number")
    departure_details: str = Field(..., description="Flight departure details like date, time and airport.")
    arrival_details: str = Field(..., description="Flight arrival details like date, time and airport.")
    duration: str = Field(..., description="Flight duration")
    price: str = Field(..., description="Price of the flight in local currency. Incase of International, provide local as well as Destination currency.")
    travel_class: str = Field(..., description="Class of the flight. Eg: Economy, Business, First, etc.")

class Hotel(BaseModel):
    name: str = Field(..., description="Name of the hotel")
    address: str = Field(..., description="Address of the hotel")
    price: str = Field(..., description="Price of the hotel stay in local currency. Incase of International, provide local as well as Destination currency.")
    image_urls: List[str] = Field(..., description="List of urls of images of the hotel")
    amenities: List[str] = Field(..., description="List of amenities provided by the hotel")
    contact: List[str] = Field(..., description="Contact details of the hotel. Eg: Phone number, Email, Website")
    rating: Optional[float] = Field(..., description="Rating of the hotel")


@CrewBase
class TravelCrew():
    """Travel crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def travelmanager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['travelmanager_agent'],
            # tools=[SerperDevTool()],
            allow_delegation=True,
            verbose=True,
            memory=True,
        )
    
    @agent
    def general_inquiry_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['general_inquiry_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
            memory=True,
        )


    @agent
    def flight_search_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['flight_search_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
            memory=True,
        )

    @agent
    def hotel_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['hotel_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
            memory=True,
        )

    @agent
    def quality_assurance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_assurance_agent'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            allow_delegation=True,
            verbose=True,
            memory=True,
        )

    @task
    def manager_task(self) -> Task:
        return Task(
            config=self.tasks_config['manager_task'],
            agent=self.travelmanager_agent(),
            # coworker=[self.flight_search_agent(), self.hotel_agent()]
        )

    @task
    def general_inquiry_task(self) -> Task:
        return Task(
            config=self.tasks_config['general_inquiry_task'],
            agent=self.general_inquiry_agent(),
            output_pydantic=Flight
        )

    @task
    def search_flights_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_flights_task'],
            agent=self.flight_search_agent(),
            output_pydantic=Flight
        )

    @task
    def search_hotels_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_hotels_task'],
            agent=self.hotel_agent(),
            output_pydantic=Hotel
        )

    @task
    def quality_assurance_review(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review'],
            agent=self.quality_assurance_agent(),
            context=[self.search_flights_task(), self.search_hotels_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Travel crew"""
        return Crew(
            # agents=self.agents,
            # tasks=self.tasks,
            agents=[self.general_inquiry_agent(), self.flight_search_agent(), self.hotel_agent(), self.quality_assurance_agent()],
            tasks=[self.general_inquiry_task(), self.search_flights_task(), self.search_hotels_task(), self.quality_assurance_review()],
            verbose=True,
            process=Process.hierarchical,
            manager_agent=self.travelmanager_agent()
            # process=Process.sequential,
            # manager_llm=llm,
            # manager_agent=None
            
        )
