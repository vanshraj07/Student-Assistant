import datetime
import os
import re
from typing import Annotated, List, Dict, Optional
from typing_extensions import TypedDict
from dataclasses import dataclass
import operator

# --- Environment Variable Loading ---
from dotenv import load_dotenv
load_dotenv()

# Environment variables
api_key = os.getenv("GOOGLE_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

if not api_key:
    print("🚨 Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

if not mongo_uri:
    print("🚨 Error: MONGO_URI not found in .env file.")
    exit(1)

# --- MongoDB Setup ---
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# MongoDB connection
try:
    client = MongoClient(mongo_uri)
    db = client.study_planner
    
    # Collections
    semester_configs = db.semester_configs
    class_schedules = db.class_schedules
    attendance_records = db.attendance_records
    calendar_events = db.calendar_events
    
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connected successfully!")
    
except ConnectionFailure as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# --- Date/Time Parsing ---
from dateutil.parser import parse as parse_date
from dateutil.rrule import rrule, DAILY, WEEKLY, MO, TU, WE, TH, FR, SA, SU

# --- Google Calendar Imports ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- LangChain & LangGraph Imports ---
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class SemesterConfig:
    start_date: str
    end_date: str
    name: str = "Current Semester"
    user_id: str = "default"

@dataclass
class ClassSchedule:
    subject: str
    days: List[str]
    start_time: str
    end_time: str
    location: str = ""
    user_id: str = "default"

@dataclass
class AttendanceRecord:
    date: str
    subject: str
    attended: bool
    notes: str = ""
    user_id: str = "default"

class StudyPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    semester_config: Optional[Dict]
    class_schedules: List[Dict]
    attendance_records: List[Dict]
    context: str

# ==============================================================================
# GOOGLE CALENDAR SETUP
# ==============================================================================
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    service = build("calendar", "v3", credentials=creds)
    return service

# ==============================================================================
# MONGODB UTILITY FUNCTIONS
# ==============================================================================

def get_semester_config(user_id="default"):
    """Get semester configuration from MongoDB"""
    try:
        config = semester_configs.find_one({"user_id": user_id})
        return config
    except Exception as e:
        logger.error(f"Error getting semester config: {e}")
        return None

def save_semester_config(config_data, user_id="default"):
    """Save semester configuration to MongoDB"""
    try:
        config_data["user_id"] = user_id
        result = semester_configs.update_one(
            {"user_id": user_id},
            {"$set": config_data},
            upsert=True
        )
        return result.acknowledged
    except Exception as e:
        logger.error(f"Error saving semester config: {e}")
        return False

def get_class_schedules(user_id="default"):
    """Get all class schedules from MongoDB"""
    try:
        schedules = list(class_schedules.find({"user_id": user_id}))
        return schedules
    except Exception as e:
        logger.error(f"Error getting class schedules: {e}")
        return []

def save_class_schedule(schedule_data, user_id="default"):
    """Save class schedule to MongoDB"""
    try:
        schedule_data["user_id"] = user_id
        result = class_schedules.insert_one(schedule_data)
        return result.acknowledged
    except Exception as e:
        logger.error(f"Error saving class schedule: {e}")
        return False

def check_duplicate_schedule(subject, start_time, end_time, days, user_id="default"):
    """Check if a class schedule already exists"""
    try:
        existing = class_schedules.find_one({
            "user_id": user_id,
            "subject": subject,
            "start_time": start_time,
            "end_time": end_time,
            "days": {"$all": days}
        })
        return existing is not None
    except Exception as e:
        logger.error(f"Error checking duplicate schedule: {e}")
        return False

def get_attendance_records(user_id="default"):
    """Get all attendance records from MongoDB"""
    try:
        records = list(attendance_records.find({"user_id": user_id}))
        return records
    except Exception as e:
        logger.error(f"Error getting attendance records: {e}")
        return []

def save_attendance_records(records_data, user_id="default"):
    """Save attendance records to MongoDB"""
    try:
        # Delete existing records for the same date to avoid duplicates
        if records_data:
            date = records_data[0]["date"]
            attendance_records.delete_many({"user_id": user_id, "date": date})
        
        # Add user_id to each record
        for record in records_data:
            record["user_id"] = user_id
        
        result = attendance_records.insert_many(records_data)
        return result.acknowledged
    except Exception as e:
        logger.error(f"Error saving attendance records: {e}")
        return False

def get_current_context(user_id="default"):
    """Get current MongoDB context to inform the agent"""
    try:
        context_parts = []
        
        # Get semester configuration
        semester_config = get_semester_config(user_id)
        if semester_config:
            start_date = parse_date(semester_config["start_date"]).strftime("%B %d, %Y")
            end_date = parse_date(semester_config["end_date"]).strftime("%B %d, %Y")
            context_parts.append(f"SEMESTER: {semester_config.get('name', 'Current Semester')} configured from {start_date} to {end_date}")
        else:
            context_parts.append("SEMESTER: Not configured yet")
        
        # Get class schedules
        schedules = get_class_schedules(user_id)
        if schedules:
            context_parts.append(f"CLASS SCHEDULES: {len(schedules)} classes configured:")
            for schedule in schedules:
                days_str = ", ".join(schedule['days'])
                location = f" in {schedule['location']}" if schedule.get('location') else ""
                context_parts.append(f"  - {schedule['subject']}: {days_str} {schedule['start_time']}-{schedule['end_time']}{location}")
        else:
            context_parts.append("CLASS SCHEDULES: None configured yet")
        
        # Get attendance summary
        records = get_attendance_records(user_id)
        if records:
            subject_stats = {}
            for record in records:
                subject = record["subject"]
                if subject not in subject_stats:
                    subject_stats[subject] = {"total": 0, "attended": 0}
                subject_stats[subject]["total"] += 1
                if record["attended"]:
                    subject_stats[subject]["attended"] += 1
            
            context_parts.append(f"ATTENDANCE: {len(records)} total records")
            for subject, stats in subject_stats.items():
                percentage = (stats["attended"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                context_parts.append(f"  - {subject}: {stats['attended']}/{stats['total']} ({percentage:.1f}%)")
        else:
            context_parts.append("ATTENDANCE: No records yet")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error getting current context: {e}")
        return "Error retrieving context from MongoDB"

def parse_days(days_str):
    """Parse day names from natural language"""
    day_mapping = {
        'monday': MO, 'mon': MO,
        'tuesday': TU, 'tue': TU, 'tues': TU,
        'wednesday': WE, 'wed': WE,
        'thursday': TH, 'thu': TH, 'thur': TH, 'thurs': TH,
        'friday': FR, 'fri': FR,
        'saturday': SA, 'sat': SA,
        'sunday': SU, 'sun': SU
    }
    
    days_lower = days_str.lower()
    weekdays = []
    day_names = []
    found_days = set()
    
    # Check full day names first
    full_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for day in full_days:
        if day in days_lower and day not in found_days:
            weekdays.append(day_mapping[day])
            day_names.append(day.capitalize())
            found_days.add(day)
    
    # Only check abbreviations if we haven't found the full name
    if not found_days:
        for day_name, day_obj in day_mapping.items():
            if day_name in days_lower and len(day_name) <= 4:
                base_day = day_name[:3] if len(day_name) <= 3 else day_name
                if base_day not in [d[:3].lower() for d in found_days]:
                    weekdays.append(day_obj)
                    full_day_name = {
                        'mon': 'Monday', 'tue': 'Tuesday', 'wed': 'Wednesday',
                        'thu': 'Thursday', 'fri': 'Friday', 'sat': 'Saturday', 'sun': 'Sunday'
                    }.get(day_name[:3], day_name.capitalize())
                    day_names.append(full_day_name)
                    found_days.add(full_day_name.lower())
    
    return weekdays, day_names

# ==============================================================================
# TOOLS
# ==============================================================================

class SemesterConfigInput(BaseModel):
    start_date: str = Field(description="Semester start date (e.g., 'January 15, 2024')")
    end_date: str = Field(description="Semester end date (e.g., 'May 30, 2024')")
    semester_name: str = Field(description="Name of the semester", default="Current Semester")

@tool(args_schema=SemesterConfigInput)
def configure_semester(start_date: str, end_date: str, semester_name: str = "Current Semester"):
    """Configure the semester dates and save to MongoDB."""
    try:
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        
        config = {
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "name": semester_name
        }
        
        if save_semester_config(config):
            return f"✅ Semester configured: {semester_name} from {start_dt.strftime('%B %d, %Y')} to {end_dt.strftime('%B %d, %Y')}"
        else:
            return "❌ Error saving semester configuration to database"
    except Exception as e:
        return f"❌ Error configuring semester: {str(e)}"

class TimetableInput(BaseModel):
    subject: str = Field(description="Subject name (e.g., 'Physics', 'Mathematics')")
    days: str = Field(description="Days of the week (e.g., 'Monday Wednesday Friday' or 'MWF')")
    start_time: str = Field(description="Class start time (e.g., '10:00 AM')")
    end_time: str = Field(description="Class end time (e.g., '11:00 AM')")
    location: str = Field(description="Classroom or location", default="")

@tool(args_schema=TimetableInput)
def add_class_schedule(subject: str, days: str, start_time: str, end_time: str, location: str = ""):
    """Add a class schedule to the timetable in MongoDB."""
    try:
        weekdays, day_names = parse_days(days)
        if not weekdays:
            return f"❌ Could not parse days: {days}"
        
        # Check for duplicates
        if check_duplicate_schedule(subject, start_time, end_time, day_names):
            return f"⚠ {subject} class already exists with same time and days. Skipping duplicate."
        
        schedule = {
            "subject": subject,
            "days": day_names,
            "start_time": start_time,
            "end_time": end_time,
            "location": location
        }
        
        if save_class_schedule(schedule):
            return f"✅ Added {subject} class: {', '.join(day_names)} from {start_time} to {end_time}"
        else:
            return "❌ Error saving class schedule to database"
    except Exception as e:
        return f"❌ Error adding class schedule: {str(e)}"

@tool
def create_semester_calendar_debug():
    """Create all semester classes as recurring events in Google Calendar with detailed debugging."""
    try:
        # Load semester config and schedules from MongoDB
        semester_config = get_semester_config()
        schedules = get_class_schedules()
        
        if not semester_config:
            return "❌ Please configure semester first using configure_semester"
        if not schedules:
            return "❌ No class schedules found. Please add classes first."
        
        debug_info = []
        debug_info.append(f"📊 MongoDB Data:")
        debug_info.append(f"Semester: {semester_config.get('name')} ({semester_config['start_date']} to {semester_config['end_date']})")
        debug_info.append(f"Schedules found: {len(schedules)}")
        for schedule in schedules:
            debug_info.append(f"  - {schedule['subject']}: {', '.join(schedule['days'])} {schedule['start_time']}-{schedule['end_time']}")
        
        service = get_calendar_service()
        
        # Parse semester dates
        start_date = parse_date(semester_config["start_date"]).date()
        end_date = parse_date(semester_config["end_date"]).date()
        
        # Check for existing events
        existing_events = []
        try:
            events_result = service.events().list(
                calendarId='primary',
                timeMin=datetime.datetime.combine(start_date, datetime.time.min).isoformat() + 'Z',
                timeMax=datetime.datetime.combine(end_date, datetime.time.max).isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            existing_events = events_result.get('items', [])
        except HttpError as e:
            debug_info.append(f"⚠ Could not fetch existing events: {e}")
        
        debug_info.append(f"\n📅 Google Calendar Data:")
        debug_info.append(f"Existing events found: {len(existing_events)}")
        
        class_events = [e for e in existing_events if e.get('summary', '').endswith(' - Class')]
        debug_info.append(f"Class events (ending with ' - Class'): {len(class_events)}")
        
        for event in class_events[:5]:  # Show first 5 class events
            summary = event.get('summary', 'No title')
            start_time = event.get('start', {}).get('dateTime', 'No time')
            debug_info.append(f"  - {summary} at {start_time}")
        
        if len(class_events) > 5:
            debug_info.append(f"  ... and {len(class_events) - 5} more class events")
        
        return "\n".join(debug_info)
        
    except Exception as e:
        return f"❌ Error in debug: {str(e)}"

@tool
def create_semester_calendar():
    """Create all semester classes as recurring events in Google Calendar, avoiding duplicates."""
    try:
        # Load semester config and schedules from MongoDB
        semester_config = get_semester_config()
        schedules = get_class_schedules()
        
        if not semester_config:
            return "❌ Please configure semester first using configure_semester"
        if not schedules:
            return "❌ No class schedules found. Please add classes first."
        
        service = get_calendar_service()
        
        # Parse semester dates
        start_date = parse_date(semester_config["start_date"]).date()
        end_date = parse_date(semester_config["end_date"]).date()
        
        # Check for existing events to avoid duplicates
        existing_events = []
        try:
            events_result = service.events().list(
                calendarId='primary',
                timeMin=datetime.datetime.combine(start_date, datetime.time.min).isoformat() + 'Z',
                timeMax=datetime.datetime.combine(end_date, datetime.time.max).isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            existing_events = events_result.get('items', [])
        except HttpError as e:
            logger.warning(f"Could not fetch existing events: {e}")
        
        # Create signatures for existing events - group by subject
        existing_signatures_by_subject = {}
        for event in existing_events:
            event_summary = event.get('summary', '')
            if event_summary and ' - Class' in event_summary:
                subject = event_summary.replace(' - Class', '')
                start_time = event.get('start', {}).get('dateTime', '')
                if start_time:
                    try:
                        event_datetime = parse_date(start_time)
                        time_signature = event_datetime.strftime('%H:%M')
                        date_signature = event_datetime.date()
                        signature = f"{date_signature}_{time_signature}"
                        
                        if subject not in existing_signatures_by_subject:
                            existing_signatures_by_subject[subject] = set()
                        existing_signatures_by_subject[subject].add(signature)
                        logger.info(f"Found existing event: {subject} on {date_signature} at {time_signature}")
                    except Exception as e:
                        logger.warning(f"Error parsing event datetime: {e}")
                        continue
        
        logger.info(f"Found {len(schedules)} schedules in MongoDB")
        for schedule in schedules:
            logger.info(f"Schedule: {schedule['subject']} - {schedule['days']} - {schedule['start_time']}")
        
        logger.info(f"Found {len(existing_events)} existing events in Google Calendar")
        logger.info(f"Existing subjects with events: {list(existing_signatures_by_subject.keys())}")
        
        created_events = 0
        skipped_events = 0
        current_date = start_date
        
        # Create events for each day
        while current_date <= end_date:
            day_name = current_date.strftime('%A').lower()
            
            for schedule in schedules:
                class_days = [day.lower() for day in schedule['days']]
                if any(day_name.startswith(day[:3]) for day in class_days):
                    
                    start_time = parse_date(schedule['start_time']).time()
                    end_time = parse_date(schedule['end_time']).time()
                    
                    time_signature = start_time.strftime('%H:%M')
                    event_signature = f"{current_date}_{time_signature}"
                    subject = schedule['subject']
                    
                    # Check if this specific subject already has an event on this date/time
                    if (subject in existing_signatures_by_subject and 
                        event_signature in existing_signatures_by_subject[subject]):
                        logger.info(f"Skipping duplicate: {subject} on {current_date} at {time_signature}")
                        skipped_events += 1
                        continue
                    
                    logger.info(f"Creating event: {subject} on {current_date} at {time_signature}")
                    
                    event_start = datetime.datetime.combine(current_date, start_time)
                    event_end = datetime.datetime.combine(current_date, end_time)
                    
                    event = {
                        'summary': f"{schedule['subject']} - Class",
                        'location': schedule.get('location', ''),
                        'description': f"Regular {schedule['subject']} class",
                        'start': {
                            'dateTime': event_start.isoformat(),
                            'timeZone': 'Asia/Kolkata',
                        },
                        'end': {
                            'dateTime': event_end.isoformat(),
                            'timeZone': 'Asia/Kolkata',
                        },
                    }
                    
                    try:
                        service.events().insert(calendarId='primary', body=event).execute()
                        created_events += 1
                        
                        # Add to existing signatures to avoid creating duplicates in same run
                        if subject not in existing_signatures_by_subject:
                            existing_signatures_by_subject[subject] = set()
                        existing_signatures_by_subject[subject].add(event_signature)
                        
                    except HttpError as create_error:
                        logger.error(f"Failed to create event for {current_date}: {create_error}")
                        continue
            
            current_date += datetime.timedelta(days=1)
        
        result_message = f"✅ Calendar sync complete!\n"
        result_message += f"📅 Created: {created_events} new events\n"
        if skipped_events > 0:
            result_message += f"⚠ Skipped: {skipped_events} duplicate events\n"
        result_message += f"🎓 Your semester calendar is up to date!"
        
        return result_message
        
    except Exception as e:
        return f"❌ Error creating calendar events: {str(e)}"
    
class AcademicEventInput(BaseModel):
    event_name: str = Field(description="Event name (e.g., 'Physics Quiz', 'Math Assignment Due')")
    date: str = Field(description="Event date (e.g., 'tomorrow', 'December 25')")
    time: str = Field(description="Event time (e.g., '2:00 PM')", default="9:00 AM")
    subject: str = Field(description="Related subject", default="")

@tool(args_schema=AcademicEventInput)
def add_academic_event(event_name: str, date: str, time: str = "9:00 AM", subject: str = ""):
    """Add a one-off academic event like quiz, assignment deadline, etc."""
    try:
        service = get_calendar_service()
        
        if date.lower() == "tomorrow":
            target_date = datetime.date.today() + datetime.timedelta(days=1)
        elif date.lower() == "today":
            target_date = datetime.date.today()
        else:
            target_date = parse_date(date).date()
        
        time_obj = parse_date(time).time()
        event_datetime = datetime.datetime.combine(target_date, time_obj)
        
        event = {
            'summary': f"{subject} - {event_name}" if subject else event_name,
            'description': f"Academic event: {event_name}",
            'start': {
                'dateTime': event_datetime.isoformat(),
                'timeZone': 'Asia/Kolkata',
            },
            'end': {
                'dateTime': (event_datetime + datetime.timedelta(hours=1)).isoformat(),
                'timeZone': 'Asia/Kolkata',
            },
        }
        
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f"✅ Added academic event: {event_name} on {event_datetime.strftime('%B %d, %Y at %I:%M %p')}\n📅 Link: {created_event.get('htmlLink')}"
        
    except Exception as e:
        return f"❌ Error adding academic event: {str(e)}"

class AttendanceInput(BaseModel):
    date: str = Field(description="Date of attendance (e.g., 'today', 'yesterday')")
    attendance_info: str = Field(description="Attendance information (e.g., 'attended all classes', 'missed Math class')")

@tool(args_schema=AttendanceInput)
def log_attendance(date: str, attendance_info: str):
    """Log daily attendance for classes in MongoDB."""
    try:
        attendance_date = parse_date(date).date().isoformat()
        
        # Load class schedules from MongoDB
        schedules = get_class_schedules()
        attendance_info_lower = attendance_info.lower()
        day_of_week = parse_date(date).strftime('%A').lower()
        
        # Find classes scheduled for this day
        todays_classes = []
        for schedule in schedules:
            schedule_days = [day.lower()[:3] for day in schedule['days']]
            if day_of_week[:3] in schedule_days:
                todays_classes.append(schedule['subject'])
        
        if not todays_classes:
            return f"No classes scheduled for {attendance_date}"
        
        new_records = []
        
        if "attended all" in attendance_info_lower or "all classes" in attendance_info_lower:
            for subject in todays_classes:
                new_records.append({
                    "date": attendance_date,
                    "subject": subject,
                    "attended": True,
                    "notes": "Present"
                })
        elif "missed all" in attendance_info_lower or "will miss all" in attendance_info_lower:
            for subject in todays_classes:
                new_records.append({
                    "date": attendance_date,
                    "subject": subject,
                    "attended": False,
                    "notes": "Absent"
                })
        else:
            for subject in todays_classes:
                if f"missed {subject.lower()}" in attendance_info_lower or f"miss {subject.lower()}" in attendance_info_lower:
                    new_records.append({
                        "date": attendance_date,
                        "subject": subject,
                        "attended": False,
                        "notes": "Absent"
                    })
                elif f"attended {subject.lower()}" in attendance_info_lower:
                    new_records.append({
                        "date": attendance_date,
                        "subject": subject,
                        "attended": True,
                        "notes": "Present"
                    })
                else:
                    new_records.append({
                        "date": attendance_date,
                        "subject": subject,
                        "attended": True,
                        "notes": "Present"
                    })
        
        if save_attendance_records(new_records):
            summary = f"✅ Logged attendance for {attendance_date}:\n"
            for record in new_records:
                status = "✓ Present" if record["attended"] else "✗ Absent"
                summary += f"  {record['subject']}: {status}\n"
            return summary
        else:
            return "❌ Error saving attendance records to database"
        
    except Exception as e:
        return f"❌ Error logging attendance: {str(e)}"

@tool
def calculate_attendance():
    """Calculate attendance percentage for each subject from MongoDB."""
    try:
        records = get_attendance_records()
        
        if not records:
            return "❌ No attendance records found."
        
        subject_stats = {}
        
        for record in records:
            subject = record["subject"]
            if subject not in subject_stats:
                subject_stats[subject] = {"total": 0, "attended": 0}
            
            subject_stats[subject]["total"] += 1
            if record["attended"]:
                subject_stats[subject]["attended"] += 1
        
        report = "📊 *Attendance Report*\n\n"
        
        for subject, stats in subject_stats.items():
            percentage = (stats["attended"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            status = "✅ Good" if percentage >= 75 else "⚠ Low" if percentage >= 65 else "❌ Critical"
            
            report += f"{subject}:\n"
            report += f"  Attended: {stats['attended']}/{stats['total']} classes\n"
            report += f"  Percentage: {percentage:.1f}% {status}\n\n"
        
        return report
        
    except Exception as e:
        return f"❌ Error calculating attendance: {str(e)}"

@tool
def view_today_schedule():
    """View today's class schedule from MongoDB."""
    try:
        schedules = get_class_schedules()
        today = datetime.date.today()
        day_name = today.strftime('%A').lower()
        
        todays_classes = []
        for schedule in schedules:
            if any(day.lower() in day_name for day in schedule['days']):
                todays_classes.append(schedule)
        
        if not todays_classes:
            return f"📅 No classes scheduled for today ({today.strftime('%A, %B %d, %Y')})"
        
        schedule_text = f"📅 *Today's Schedule* ({today.strftime('%A, %B %d, %Y')}):\n\n"
        
        for class_info in sorted(todays_classes, key=lambda x: parse_date(x['start_time']).time()):
            schedule_text += f"🕐 *{class_info['subject']}*\n"
            schedule_text += f"   Time: {class_info['start_time']} - {class_info['end_time']}\n"
            if class_info.get('location'):
                schedule_text += f"   Location: {class_info['location']}\n"
            schedule_text += "\n"
        
        return schedule_text
        
    except Exception as e:
        return f"❌ Error viewing schedule: {str(e)}"

# ==============================================================================
# LANGGRAPH AGENT SETUP
# ==============================================================================

tools = [
    configure_semester,
    add_class_schedule,
    create_semester_calendar,
    create_semester_calendar_debug,
    add_academic_event,
    log_attendance,
    calculate_attendance,
    view_today_schedule
]

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key=api_key
)
model = model.bind_tools(tools)

def should_continue(state: StudyPlannerState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    return "end"

def call_model(state: StudyPlannerState):
    # Get current MongoDB context before processing
    current_context = get_current_context()
    
    system_message = f"""You are a Study Planner Assistant that helps students manage their academic schedule using Google Calendar and MongoDB.

Your capabilities:
1. *Semester Setup*: Configure semester start/end dates (stored in MongoDB)
2. *Timetable Management*: Parse and add class schedules from natural language (stored in MongoDB)
3. *Calendar Integration*: Create recurring events, avoid holidays
4. *Academic Events*: Add quizzes, assignments, deadlines
5. *Attendance Tracking*: Log daily attendance and calculate percentages (stored in MongoDB)
6. *Schedule Viewing*: Show today's classes

All data is persistently stored in MongoDB, so it will be available across sessions.

IMPORTANT: Before responding, always check the current MongoDB data below to see what's already configured:

{current_context}

Based on the above data:
- If semester is already configured, don't ask for it again
- If class schedules exist, use them directly for calendar creation
- If attendance records exist, reference them in your responses
- Always acknowledge existing data when relevant

Always be helpful, clear, and encourage good attendance habits. When students have low attendance, gently remind them about the importance of regular attendance."""

    messages = [AIMessage(content=system_message)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Create the graph
workflow = StateGraph(StudyPlannerState)
workflow.add_node("agent", call_model)
tool_node = ToolNode(tools)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", 
    should_continue, 
    {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")

app = workflow.compile()

print("✅ Study Planner Assistant with MongoDB Context is ready!")
print("🎓 You can now manage your semester, classes, and attendance.")
print("💾 All data is persistently stored in MongoDB and checked automatically.")
print("\nExample commands:")
print("- 'Configure my semester from January 15 to May 30, 2024'")
print("- 'Add Physics class on Monday Wednesday Friday from 10 AM to 11 AM'")
print("- 'Create all my semester events in calendar'")
print("- 'I attended all my classes today'")
print("- 'Show my attendance report'")

# For LangSmith integration
if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="Hello! I'm ready to set up my study planner with MongoDB.")],
        "semester_config": None,
        "class_schedules": [],
        "attendance_records": [],
        "context": ""
    })
    print("\n" + "="*50)
    print("ASSISTANT RESPONSE:")
    print("="*50)
    print(result["messages"][-1].content)