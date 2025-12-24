import anthropic
import os
import json
import base64
from typing import List, Dict, Optional
from fastapi import UploadFile
from serpapi import GoogleSearch

from dotenv import load_dotenv
load_dotenv()
import requests

class ClaudeConversation:
    
    async def google_search(self,query, start=0):
        num_results = 10
        api_key = "AIzaSyDGUJz3wavssYikx5wDq0AcD2QlRt4vS5c"
        cx = "650310331e0e3490e"
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(num_results, 10)
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return []
        
        data = response.json()
        results = []
        
        for item in data.get("items", []):
            if item.get("link"):
                results.append({
                    "url": item["link"],
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")
                })
        
        return results
    
        results = []    
        try:        
            params = {
                "engine": "google",
                "q": query,
                "api_key": "c65412e924d81ecb726a7c013ae0f04897bc8d069e8acadc5a085d9198e64d22",
                "num": 10,
                "start": start,
            }
            
            search = GoogleSearch(params)
            search_dict = search.get_dict()
            items = search_dict.get("organic_results")
            
            if not items:
                print(f"[DEBUG] No search results found")
                return results
                
            for item in items:
                if item.get("link"):
                    res = {
                        "url": item["link"], 
                        "snippet": item.get("snippet", ""),
                        "title": item.get("title", "")
                    }
                    results.append(res)
            
            print(f"[DEBUG] Found {len(results)} search results")
    
        except Exception as e:
            print(f"[DEBUG] Search error: {str(e)}")
        
        return results

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-20250514", messages: Optional[List[Dict]] = None):
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY",""),timeout=1800)        
        
        self.model = model
        self.max_tokens = 16000
        self.thinking_budget = 10000
        
        if self.max_tokens <= self.thinking_budget:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be greater than "
                f"thinking_budget ({self.thinking_budget})"
            )
        
        self.messages: List[Dict] = messages if messages is not None else []
        self._last_stop_reason = None
    
    def set_token_limits(self, max_tokens: int, thinking_budget: int):
        if max_tokens <= thinking_budget:
            raise ValueError(
                f"max_tokens ({max_tokens}) must be greater than "
                f"thinking_budget ({thinking_budget})"
            )
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
    
    async def _process_files(self, files: List[UploadFile]) -> List[Dict]:
        """Process uploaded files and convert to Anthropic API format"""
        file_contents = []
        
        for file in files:
            content_type = file.content_type
            file_data = await file.read()
            
            if content_type.startswith('image/'):
                # Handle images
                base64_data = base64.b64encode(file_data).decode('utf-8')
                file_contents.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content_type,
                        "data": base64_data
                    }
                })
            elif content_type == 'application/pdf':
                # Handle PDFs
                base64_data = base64.b64encode(file_data).decode('utf-8')
                file_contents.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_data
                    }
                })
            elif content_type.startswith('text/'):
                # Handle text files
                text_content = file_data.decode('utf-8')
                file_contents.append({
                    "type": "text",
                    "text": f"File: {file.filename}\n\n{text_content}"
                })
        
        return file_contents
    
    async def _generate_search_query(self, user_message: str):
        """Fast LLM call to generate Google search query using only user prompts"""
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        user_prompts = [msg for msg in self.messages if msg["role"] == "user"]
        
        query_prompt = f"""Based on the user's previous conversation history and their latest question, determine if a web search is needed and generate a search query if required.

Previous user messages:
{[msg["content"] for msg in user_prompts]}

Current user question: "{user_message}"

First, decide if web search is needed:
- YES: For current information (weather, stock prices, news, sports scores, etc.)
- NO: For general knowledge, explanations, calculations, creative tasks, or conversational responses

Response format (respond with ONLY these two lines):
SEARCH_NEEDED: YES or NO
QUERY: [your search query here, or NONE if not needed]

Rules for search query (if needed):
- Search query MUST be grammatically correct and meaningful
- Make the query independently searchable on Google with NO assumed context
- For time-sensitive queries, include the current date or year

Generate the response:"""
        
        query_messages = [{"role": "user", "content": query_prompt}]
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=f"Today's date is {current_date}. Use this date when generating search queries for current information.",
            messages=query_messages
        )
        
        if response.content and len(response.content) > 0:
            response_text = response.content[0].text.strip()
            
            lines = response_text.split('\n')
            search_needed = False
            query = None
            
            for line in lines:
                if line.startswith('SEARCH_NEEDED:'):
                    search_needed = 'YES' in line.upper()
                elif line.startswith('QUERY:'):
                    query = line.replace('QUERY:', '').strip()
                    if query.upper() == 'NONE':
                        query = None
            
            return {"search_needed": search_needed, "query": query}
        
        return {"search_needed": False, "query": None}
    
    async def send_message(self, user_message: str, files: Optional[List[UploadFile]] = None, max_iterations: int = 5, simple_search = True):
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        if simple_search:
            # First: Generate search query quickly using only user prompts
            search_info = await self._generate_search_query(user_message)
            if search_info["search_needed"] and search_info["query"]:
                yield {"type": "search_query", "text": search_info["query"]}
                    
        # Process uploaded files
        message_content = []
        if files:
            file_contents = await self._process_files(files)
            message_content.extend(file_contents)
        
        # Add text message
        message_content.append({
            "type": "text",
            "text": user_message
        })
        
        # Add user message to history
        self.messages.append({"role": "user", "content": message_content})
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            try:
                assistant_content = []
                
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=f"Today's date is {current_date}. Use this date when providing current information.",
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    },
                    tools=[
                        {
                            "type": "web_search_20250305",
                            "name": "web_search"
                        }
                    ],
                    messages=self.messages
                ) as stream:
                    async for event in stream:
                        if hasattr(event, 'type'):
                            if event.type == "content_block_delta":
                                if hasattr(event, 'delta'):
                                    if hasattr(event.delta, 'thinking'):
                                        yield {"type": "thinking", "text": event.delta.thinking}
                                    elif hasattr(event.delta, 'text'):
                                        yield {"type": "content", "text": event.delta.text}
                    
                    final_message = await stream.get_final_message()
                    self._last_stop_reason = final_message.stop_reason
                    
                    for block in final_message.content:
                        if block.type == "text":
                            assistant_content.append({
                                "type": "text",
                                "text": block.text
                            })
                
                history_content = [
                    block for block in assistant_content 
                    if block.get("type") != "thinking"
                ]
                
                self.messages.append({
                    "role": "assistant",
                    "content": history_content
                })
                break
            
            except Exception as e:
                raise
    
    def get_history(self):
        return self.messages.copy()
    
    def clear_history(self):
        self.messages = []
    
    def save_conversation(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.messages, f, indent=2)
    
    def load_conversation(self, filename: str):
        with open(filename, 'r') as f:
            self.messages = json.load(f)


if __name__ == "__main__":
    import asyncio
    
    async def test_conversation():
        print("\n" + "=" * 80)
        print("TEST 1: NEW CONVERSATION WITH MULTIPLE TURNS")
        print("=" * 80)
        
        conversation = ClaudeConversation()
        
        print("\n👤 YOU: What is Microsoft (MSFT) stock price right now?")
        print("-" * 80)
        async for chunk in conversation.send_message("What is Microsoft (MSFT) stock price right now?"):
            if chunk["type"] == "search_query":
                print(f"🔍 [SEARCH QUERY] {chunk['text']}")
        
        print("\n👤 YOU: How does that compare to yesterday's closing price?")
        print("-" * 80)
        async for chunk in conversation.send_message("How does that compare to yesterday's closing price?"):
            if chunk["type"] == "search_query":
                print(f"🔍 [SEARCH QUERY] {chunk['text']}")
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
    
    asyncio.run(test_conversation())