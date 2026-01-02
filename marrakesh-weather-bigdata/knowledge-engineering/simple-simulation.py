import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# --- 1. THE ONTOLOGY (Defining Concepts) ---
# This satisfies: "Extraction and modeling of knowledge" [cite: 18]

@dataclass
class MeteoFact:
    """Represents a single day of meteorological data."""
    date: str
    tmax: float  # Maximum Temperature
    rhum: float  # Relative Humidity
    wspd: float  # Wind Speed

@dataclass
class ClimateEvent:
    """Represents an inferred event (The Output)."""
    event_type: str  # e.g., "Heatwave", "High Risk"
    severity: str    # "Moderate", "High", "Critical"
    description: str
    date: str

# --- 2. THE RULE BASE (Defining Logic) ---
# This satisfies: "Identify thresholds and relations" 

class HeatwaveRules:
    """
    The Rule Engine that holds the logic for detecting heatwaves.
    """
    
    # Thresholds defined based on domain knowledge (or your doc examples)
    TEMP_CRITICAL = 42.0  # As mentioned in doc 
    TEMP_HIGH = 38.0
    HUMIDITY_DANGER = 20.0 # Low humidity + High heat = dry heat danger
    
    @staticmethod
    def check_daily_risk(fact: MeteoFact) -> Optional[ClimateEvent]:
        """
        Rule 1: Immediate Daily Check (Simple Logic)
        Checks if a single day is dangerous.
        """
        # Rule A: Critical Heat (Doc example logic)
        if fact.tmax >= HeatwaveRules.TEMP_CRITICAL:
            return ClimateEvent(
                event_type="Heatwave Alert",
                severity="Critical",
                description=f"Temp ({fact.tmax}°C) exceeds critical threshold of {HeatwaveRules.TEMP_CRITICAL}°C.",
                date=fact.date
            )
        
        # Rule B: Dry Heat Combo (Complex Logic)
        # If temp is high AND humidity is very low (dry heat)
        elif fact.tmax >= HeatwaveRules.TEMP_HIGH and fact.rhum < HeatwaveRules.HUMIDITY_DANGER:
            return ClimateEvent(
                event_type="Dry Heat Risk",
                severity="High",
                description=f"High heat ({fact.tmax}°C) with dangerous low humidity ({fact.rhum}%).",
                date=fact.date
            )
            
        return None

    @staticmethod
    def check_persistent_heatwave(history: List[MeteoFact]) -> Optional[ClimateEvent]:
        """
        Rule 2: Temporal Logic (The "3 Days" Rule) 
        'If Temperature > 42°C for 3 days then Heatwave'
        """
        if len(history) < 3:
            return None
            
        # Get last 3 days
        last_3_days = history[-3:]
        
        # Check if ALL 3 days exceeded the threshold
        # This is the "Inference" step
        if all(day.tmax > HeatwaveRules.TEMP_CRITICAL for day in last_3_days):
            return ClimateEvent(
                event_type="Persistent Heatwave",
                severity="EXTREME",
                description="Temperature has exceeded 42°C for 3 consecutive days.",
                date=last_3_days[-1].date
            )
        return None

# --- 3. EXECUTION (Simulation) ---

# Mock Data (Simulation of Meteostat input)
data_stream = [
    MeteoFact(date="2026-07-10", tmax=35.0, rhum=40.0, wspd=10.0), # Normal
    MeteoFact(date="2026-07-11", tmax=43.0, rhum=15.0, wspd=5.0),  # Critical Day
    MeteoFact(date="2026-07-12", tmax=44.0, rhum=12.0, wspd=8.0),  # Critical Day
    MeteoFact(date="2026-07-13", tmax=45.5, rhum=10.0, wspd=2.0),  # Critical Day -> SHOULD TRIGGER 3-DAY RULE
]

print("--- STARTING KNOWLEDGE ENGINE ANALYSIS ---")

history_buffer = []

for day_data in data_stream:
    print(f"\nProcessing Date: {day_data.date} | TMax: {day_data.tmax}°C")
    
    # 1. Update History
    history_buffer.append(day_data)
    
    # 2. Apply Daily Rules
    daily_alert = HeatwaveRules.check_daily_risk(day_data)
    if daily_alert:
        print(f"   [!] ALERT GENERATED: {daily_alert.event_type} - {daily_alert.description}")
    
    # 3. Apply Temporal Rules (The complex logic)
    temporal_alert = HeatwaveRules.check_persistent_heatwave(history_buffer)
    if temporal_alert:
        print(f"   [!!!] COMPLEX EVENT DETECTED: {temporal_alert.event_type}")
        print(f"         Reason: {temporal_alert.description}")
