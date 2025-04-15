from datetime import date
from enum import Enum
from typing import List, Optional

from extended_dataset_profile import ExtendedDatasetProfile as ExtendedDatasetProfile
from pydantic import BaseModel, Field, model_serializer


class Config(BaseModel):
    """
    Extended dataset profile service configuration

    This configuration contains all customizable variables for the analysis of assets.
    All analyzer configurations are collected here.
    """


class DataCategory(str, Enum):
    TrafficInformation = "Traffic Information"
    RoadworksAndRoadConditions = "Roadworks and Road Conditions"
    TrafficFlowInformation = "Traffic Flow Information"
    ParkingInformation = "Parking Information"
    Electromobility = "Electromobility"
    TrafficSignsAndSpeedInformation = "Traffic Signs and Speed Information"
    WeatherInformation = "Weather Information"
    PublicTransportInformation = "Public Transport Information"
    SharedAndOnDemandMobility = "Shared and On-Demand Mobility"
    InfrastructureAndLogistics = "Infrastructure and Logistics"
    Various = "Various"


class DataSubCategory(str, Enum):
    Accidents = "Accidents"
    HazardWarnings = "Hazard Warnings"
    Roadworks = "Roadworks"
    RoadConditions = "Road Conditions"
    RealtimeTrafficFlowData = "Realtime Traffic Flow Data"
    ForecastTrafficFlowData = "Forecast Traffic Flow Data"
    AvailabilityAndForecast = "Availability and Forecast"
    Prices = "Prices"
    AvailabilityOfChargingStation = "Availability of Charging Station"
    LocationOfChargingStation = "Location of Charging Station"
    PricesOfChargingStation = "Prices of Charging Station"
    DynamicSpeedInformation = "Dynamic Speed Information"
    DynamicTrafficSigns = "Dynamic Traffic Signs"
    StaticTrafficSigns = "Static Traffic Signs"
    CurrentWeatherConditions = "Current weather conditions"
    WeatherForecast = "Weather Forecast"
    SpecialEventsOrDisruptions = "Special Events or Disruptions"
    Timetables = "Timetables"
    Fare = "Fare"
    LocationInformation = "Location Information"
    VehicleInformation = "Vehicle information"
    Availability = "Availability"
    Location = "Location"
    Range = "Range"
    GeneralInformationAboutPlanningOfRoutes = "General Information About Planning Of Routes"
    PedestrianNetworks = "Pedestrian Networks"
    CyclingNetworks = "Cycling Networks"
    RoadNetwork = "Road Network"
    WaterRoutes = "Water Routes"
    CargoAndLogistics = "Cargo and Logistics"
    TollInformation = "Toll Information"


class TransportMode(str, Enum):
    Road = "Road"
    Rail = "Rail"
    Water = "Water"
    Air = "Air"


class DateRange(BaseModel):
    start: date
    end: date

    @model_serializer(when_used="json")
    def _serialize_json(self) -> str:
        FORMAT = "%d.%m.%Y"
        return f"{self.start.strftime(FORMAT)} - {self.end.strftime(FORMAT)}"


TemporalCoverage = None | date | DateRange


class MobilityDataSpaceOntology(BaseModel):
    """
    This is the mobility data space ontology.

    Details:
        It encodes information about mobility relevant data sets.
        Status: Jul 17, 2024.

    URL:
        https://github.com/Mobility-Data-Space/mobility-data-space/wiki/MDS-Asset-Attributes
    """

    name: str = Field(description="Dataset Name", examples=["Traffic situation in Hamburg"], serialization_alias="Name")
    version: Optional[str] = Field(
        default=None, description="Dataset Version", examples=["1.0.0"], serialization_alias="Version"
    )
    asset_id: str = Field(
        description="Dataset Id (in the MDS Frontend will be generated automatically)",
        examples=["traffic-situation-in-hamburg-1.0"],
        serialization_alias="Asset ID",
    )
    description: Optional[str] = Field(
        default=None,
        description="Dataset Description",
        examples=[
            "The dataset contains the traffic situation in real time on the Hamburg road network. The traffic situation is divided into 4 condition classes: flowing traffic (green), heavy traffic (orange), slow-moving traffic (red), queued traffic (dark red)."
        ],
        serialization_alias="Description",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords describing the dataset",
        examples=[["traffic", "hamburg"], ["traffic jams"]],
        serialization_alias="Keywords",
    )
    languages: Optional[str] = Field(
        default=None,
        description='Language of the dataset (use "Multilingual" if more than one language is used)',
        examples=["German"],
        serialization_alias="Language",
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Content type of the dataset",
        examples=["application/json"],
        serialization_alias="Content type",
    )
    endpoint_documentation: Optional[str] = Field(
        default=None,
        description="Documentation describing the dataset, its parameters and values",
        examples=["https://api.hamburg.de/datasets/v1/verkehrslage//api?f=html"],
        serialization_alias="Endpoint Documentation",
    )
    publisher: Optional[str] = Field(
        default=None,
        description="Homepage of the participant who makes the dataset available within the MDS",
        examples=["https://mobility-dataspace.eu/"],
        serialization_alias="Publisher",
    )
    organization: str = Field(
        description="Legal name of the participant who makes the dataset available within the MDS",
        examples=["DRM GmbH"],
        serialization_alias="Organization",
    )
    standard_licence: Optional[str] = Field(
        default=None,
        description="License under which is the dataset available",
        examples=["https://www.govdata.de/dl-de/by-2-0"],
        serialization_alias="Standard licence",
    )
    data_category: DataCategory = Field(
        description="Vordefined MDS Data Category",
        examples=["Traffic Flow Information"],
        serialization_alias="Data Category",
    )
    data_subcategory: Optional[DataSubCategory] = Field(
        default=None,
        description="Vordefined MDS Data Subcategory",
        examples=["Realtime Traffic Flow Data"],
        serialization_alias="Data Subcategory",
    )
    data_model: Optional[str] = Field(
        default=None,
        description="Mobility Data Standard",
        examples=["DATEX II", "TPEG", "Proprietary"],
        serialization_alias="Data Model",
    )
    transport_mode: Optional[TransportMode] = Field(
        default=None, description="Vordefined Transport Mode", examples=["Road"], serialization_alias="Transport Mode"
    )
    geo_reference_model: Optional[str] = Field(
        default=None,
        description="Geo Referencing Method",
        examples=["OpenLR", "GeoJSON"],
        serialization_alias="Geo Reference Model",
    )
    sovereign: Optional[str] = Field(
        default=None,
        description="Legal name of the owner of the dataset",
        examples=["LGV Hamburg"],
        serialization_alias="Sovereign",
    )
    data_update_frequency: Optional[str] = Field(
        default=None,
        description="How often is the dataset updated.",
        examples=["Every 5 min."],
        serialization_alias="Data update frequency",
    )
    geo_location: Optional[str] = Field(
        default=None,
        description="Simple description of the relevant geolocation.",
        examples=["Hamburg and vicinity"],
        serialization_alias="Geo location",
    )
    nuts_location: Optional[str] = Field(
        default=None,
        description="NUTS code(s) for the relevant geolocation.",
        examples=["DE60"],
        serialization_alias="NUTS location",
    )
    data_samples: Optional[str] = Field(
        default=None, description="Dataset samples if available", serialization_alias="Data samples"
    )
    reference_files: Optional[str] = Field(
        default=None, description="Dataset schemas or other references", serialization_alias="Reference files"
    )
    temporal_coverage: TemporalCoverage = Field(
        default=None,
        description="Start and/or end date for the dataset",
        examples=["14.05.2024 - 14.05.2024"],
        serialization_alias="Temporal coverage",
    )
    condition_for_use: Optional[str] = Field(
        default=None,
        description="Additional condiotions for use, source reference, copyright etc.",
        examples=["Source reference: Freie und Hansestadt Hamburg, Behörde für Verkehr und Mobilitätswende"],
        serialization_alias="Condition for use",
    )
