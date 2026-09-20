import json
from rag_pipeline.conversation.llm import client, MODEL

def check_booking_intent_and_extact(message:str): 
    prompt = f"""You are analyzing a user's message to detect if they are trying to book an interview.

Message: "{message}"

Determine if this message contains a request to book or schedule an interview AND includes enough information to do so:
- name
- email address
- date
- time

Return ONLY a valid JSON object.

The JSON must contain exactly these five fields:

{{
    "is_booking_request": true,
    "name": null,
    "email": null,
    "date": null,
    "time": null
}}

Use true or false for "is_booking_request".
Use a string for name, email, date, and time when the information is present.
Use null when a field is missing or unclear.

If the message is not about booking an interview, return:
{{
    "is_booking_request": false,
    "name": null,
    "email": null,
    "date": null,
    "time": null
}}
"""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw_output = completion.choices[0].message.content
    parsed = json.loads(raw_output)

    if not parsed.get("is_booking_request"):
        return None

    if not all([parsed.get("name"), parsed.get("email"), parsed.get("date"), parsed.get("time")]):
        return None

    return {
        "name": parsed["name"],
        "email": parsed["email"],
        "date": parsed["date"],
        "time": parsed["time"],
    }