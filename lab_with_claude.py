import anthropic
import json
from typing import List, Dict, Optional
from fastapi import UploadFile
import asyncio
from datetime import datetime
from tables_scraper import scrape_tables_parallel, BrowserPool

# ✅ Global browser pool (shared across requests)
_global_browser_pool: Optional[BrowserPool] = None
_pool_lock = asyncio.Lock()

async def get_browser_pool():
    """Get or create global browser pool"""
    global _global_browser_pool
    
    async with _pool_lock:
        if _global_browser_pool is None or not _global_browser_pool.initialized:
            _global_browser_pool = BrowserPool(pool_size=2, max_tabs_per_browser=10)
            await _global_browser_pool.initialize()
    
    return _global_browser_pool


async def classify_web_search_needed(query: str, client, model: str) -> dict:
    """
    Classifier for first message: What should we do?
    Returns: {"action": "conversation|create_app", "use_web_search": bool}
    """
    
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    prompt = f"""The current date is {current_date}.

User's first message: "{query}"

Determine what action to take:

1. If this is a CASUAL/GREETING message (hi, hello, how are you, etc.):
   - action: "conversation"
   - use_web_search: false

2. If this is a RESEARCH/APP REQUEST:
   - action: "create_app"
   - use_web_search: true/false (based on if it needs current data)

Return ONLY valid JSON:
{{"action": "conversation|create_app", "use_web_search": true|false}}"""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        print(f"🔍 First message action: {result.get('action')}, Web search: {result.get('use_web_search', False)}")
        return result
    
    except Exception as e:
        print(f"⚠️ Classifier error: {e}, defaulting to conversation")
        return {"action": "conversation", "use_web_search": False}
    
async def classify_followup_message(
    conversation_messages: List[Dict], 
    user_message: str, 
    client, 
    model: str,
    has_existing_html: bool
) -> Dict[str, any]:
    """
    Full classifier for follow-up messages
    Returns: {"route": "conversation|create_app|update_app", "use_web_search": bool}
    """
    
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # Format conversation history
    history_text = ""
    for msg in conversation_messages[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
        history_text += f"{role}: {content[:500]}...\n\n"
    
    existing_html_context = ""
    if has_existing_html:
        existing_html_context = "\n\nNOTE: There is an existing HTML app that can be updated."
    
    router_prompt = f"""The current date is {current_date}.

Analyze the conversation and classify the user's message.

Recent conversation history:
{history_text}{existing_html_context}

Latest user message: "{user_message}"

Classify the message:

1. ROUTE (choose one):

   a) "conversation" - Just answering questions, NO app creation/update
      - Asking about existing content
      - Clarifying questions
      - Discussion
   
   b) "create_app" - Create NEW app (topic is significantly DIFFERENT from current conversation)
      - New topic unrelated to current conversation
      - User wants to research something completely different
      - Examples: Currently discussing "Apple" → User asks "Research climate change"
   
   c) "update_app" - Update EXISTING app (topic is RELATED to current conversation)
      - Extending/modifying current topic
      - Adding data to existing research
      - Styling changes
      - Examples: Currently discussing "Apple vs Microsoft" → User asks "Add revenue data"
      - ONLY if existing HTML app is present

2. USE_WEB_SEARCH (true/false):
   
   true = Need current/factual data from web
   - Current statistics, prices, news
   - Comparative analysis with real numbers
   - Time-sensitive info ("2025", "latest", "current")
   
   false = LLM knowledge sufficient
   - General knowledge topics
   - Creative/educational content
   - Conceptual explanations
   - Styling changes ("make it dark mode")

CRITICAL RULES:
- If no existing HTML app → cannot choose "update_app"
- "conversation" never needs web search (always false)
- New unrelated topic → "create_app"
- Related/extending topic + existing HTML → "update_app"

Return ONLY valid JSON:
{{"route": "conversation|create_app|update_app", "use_web_search": true|false, "reasoning": "..."}}"""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": router_prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        route = result.get("route", "conversation")
        use_web_search = result.get("use_web_search", False)
        
        # Fallback: if update_app but no existing HTML, change to create_app
        if route == "update_app" and not has_existing_html:
            print(f"⚠️ Router chose 'update_app' but no existing HTML, changing to 'create_app'")
            route = "create_app"
        
        # Conversation never uses web search
        if route == "conversation":
            use_web_search = False
        
        print(f"🔀 Router: {route}, Web search: {use_web_search} - {result.get('reasoning', '')}")
        return {"route": route, "use_web_search": use_web_search}
    
    except Exception as e:
        print(f"⚠️ Classifier error: {e}, defaulting to conversation")
        return {"route": "conversation", "use_web_search": False}


class DeepResearch:
    """
    Intelligent research system with multi-stage deep research and smart routing
    """
    
    def __init__(self, conversation):
        """
        Args:
            conversation: ClaudeConversation instance
        """
        self.conversation = conversation
        self.client = conversation.client
        self.model = conversation.model
        self.model = "claude-sonnet-4-20250514"
        self.html_model = "claude-opus-4-20250514"
        self.browser_pool = None
        self.query_tables = {}
        self.last_research_data = None
    
    async def research(
    self, 
    query: str, 
    files: Optional[List[UploadFile]] = None,
    existing_html: Optional[str] = None
):
        """
        Main entry point with intelligent routing
        
        Args:
            query: User's message/question
            files: Optional uploaded files
            existing_html: Optional existing HTML app to update
        
        Yields: {"type": "search_query"/"sources"/"tables"/"reasoning"/"content"/"html_app"/"research_summary"}
        """
        
        # ✅ Detect if this is a follow-up
        follow_up = len(self.conversation.messages) > 0
        
        if not follow_up:
            # ✅ FIRST MESSAGE - Classify action
            yield {"type": "reasoning", "text": "🔍 Analyzing your request..."}
            
            decision = await self._classify_first_message(query)
            
            if decision["action"] == "conversation":
                # Just greeting/casual conversation
                yield {"type": "reasoning", "text": "💬 Hello! How can I help you today?"}
                async for chunk in self.conversation.send_message(query, files, simple_search=False):
                    yield chunk
            else:
                # Create app
                async for chunk in self._create_app_flow(query, files, decision["use_web_search"]):
                    yield chunk
        
        else:
            # ✅ FOLLOW-UP MESSAGE - Classify route + web search
            decision = await classify_followup_message(
                self.conversation.messages,
                query,
                self.client,
                self.model,
                has_existing_html=existing_html is not None
            )
            
            route = decision["route"]
            use_web_search = decision["use_web_search"]
            
            if route == "conversation":
                # Just answer, no app
                yield {"type": "reasoning", "text": "💬 Answering your question..."}
                async for chunk in self.conversation.send_message(query, files, simple_search=False):
                    yield chunk
            
            elif route == "create_app":
                # New topic - create new app
                async for chunk in self._create_app_flow(query, files, use_web_search):
                    yield chunk
            
            elif route == "update_app":
                # Update existing app
                async for chunk in self._update_app_flow(query, files, existing_html, use_web_search):
                    yield chunk


    async def _classify_first_message(self, query: str) -> dict:
        """
        Classifier for first message: conversation or create_report?
        Returns: {"action": "conversation|create_report", "use_web_search": bool}
        """
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        prompt = f"""The current date is {current_date}.

User's first message: "{query}"

Determine what action to take:

1. If this is a CASUAL/GREETING message:
- Examples: "hi", "hello", "how are you", "hey there", "good morning"
- action: "conversation"
- use_web_search: false

2. If this is a REQUEST FOR RESEARCH/REPORT:
- Examples: "Research Apple vs Microsoft", "Analyze climate change", "Report on AI trends"
- action: "create_report"
- use_web_search: Decide based on below
- If its not an explicit attempt of making a conversation or a greeting - this MUST always result in creaton of a WEBSITE.

For create_report, determine if web search is needed:

Web search NEEDED (true):
- Current data, statistics, prices, news
- Comparative analysis with real numbers
- Time-sensitive information (2024, 2025, "latest", "current")

LLM knowledge SUFFICIENT (false):
- General knowledge topics ("types of cheese", "history of Rome")
- Creative/educational content
- Conceptual explanations

Return ONLY valid JSON:
{{"action": "conversation|create_report", "use_web_search": true|false}}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            action = result.get("action", "conversation")
            use_web_search = result.get("use_web_search", False)
            
            print(f"🔍 First message - Action: {action}, Web search: {use_web_search}")
            return {"action": action, "use_web_search": use_web_search}
        
        except Exception as e:
            print(f"⚠️ Classifier error: {e}, defaulting to conversation")
            return {"action": "conversation", "use_web_search": False}

        
    async def _create_app_flow(self, query: str, files: Optional[List[UploadFile]], use_web_search: bool):
        """
        Create new app with multi-stage deep research
        WITH or WITHOUT web search based on flag
        """
        
        yield {"type": "reasoning", "text": "🔍 Starting deep research process..."}
        
        self.browser_pool = await get_browser_pool()
        self.query_tables = {}
        
        # Phase 1: Generate Level 1 queries
        yield {"type": "reasoning", "text": "📊 Analyzing research question and generating main branches..."}
        level1_queries = await self._generate_level1_queries(query)
        yield {"type": "reasoning", "text": f"✅ Generated {len(level1_queries)} research branches"}
        
        if use_web_search:
            yield {"type": "reasoning", "text": "🔎 Beginning web searches for Level 1 queries..."}
            
            for i, q in enumerate(level1_queries, 1):
                search_info = await self.conversation._generate_search_query(q)
                if search_info["search_needed"] and search_info["query"]:
                    yield {"type": "search_query", "text": search_info["query"]}
                    
                    search_results = await self.conversation.google_search(search_info["query"])
                    yield {"type": "sources", "content": search_results}
                    
                    yield {"type": "reasoning", "text": f"📄 Extracting tables from Branch {i} sources..."}
                    
                    urls = [result["url"] for result in search_results[:5]]
                    if urls:
                        tables = await self._extract_tables_from_urls(urls)
                        if tables:
                            self.query_tables[q] = tables
                            yield {"type": "tables", "content": tables}
                            yield {"type": "reasoning", "text": f"✅ Extracted {len(tables)} tables from Branch {i}"}
        
        yield {"type": "reasoning", "text": "🌳 Expanding branches into detailed sub-queries..."}
        
        # Phase 2: Generate Level 2 queries
        level2_queries = {}
        for i, l1_query in enumerate(level1_queries, 1):
            l2_queries = await self._generate_level2_queries(l1_query, query)
            level2_queries[l1_query] = l2_queries
            
            yield {"type": "reasoning", "text": f"✅ Branch {i} expanded into {len(l2_queries)} sub-queries"}
            
            if use_web_search:
                for j, q in enumerate(l2_queries, 1):
                    search_info = await self.conversation._generate_search_query(q)
                    if search_info["search_needed"] and search_info["query"]:
                        yield {"type": "search_query", "text": search_info["query"]}
                        
                        search_results = await self.conversation.google_search(search_info["query"])
                        yield {"type": "sources", "content": search_results}
                        
                        urls = [result["url"] for result in search_results[:5]]
                        if urls:
                            tables = await self._extract_tables_from_urls(urls)
                            if tables:
                                self.query_tables[q] = tables
                                yield {"type": "tables", "content": tables}
        
        total_queries = sum(len(queries) for queries in level2_queries.values())
        yield {"type": "reasoning", "text": f"✅ Data collection complete: {total_queries} queries executed"}
        
        # Phase 3: Build research data structure
        yield {"type": "reasoning", "text": "📦 Organizing research data..."}
        
        research_data = {
            "query": query,
            "use_web_search": use_web_search,
            "branches": []
        }
        
        for i, l1_query in enumerate(level1_queries, 1):
            branch_data = {
                "title": l1_query,
                "sub_queries": []
            }
            
            l2_queries = level2_queries.get(l1_query, [])
            for j, l2_query in enumerate(l2_queries, 1):
                tables = self.query_tables.get(l2_query, [])
                
                sub_query_data = {
                    "question": l2_query,
                    "tables": tables if use_web_search else [],
                    "tables_count": len(tables) if use_web_search else 0
                }
                branch_data["sub_queries"].append(sub_query_data)
            
            research_data["branches"].append(branch_data)
        
        self.last_research_data = research_data
        
        yield {"type": "reasoning", "text": "✅ Research data organized"}
        
        # Phase 4: Generate HTML app
        yield {"type": "reasoning", "text": "🎨 Generating interactive HTML research website..."}
        
        html_app = await self._create_html_app(research_data)
        
        yield {"type": "html_app", "content": html_app}
        
        # Phase 5: ALWAYS Generate Research Summary & Methodology
        yield {"type": "reasoning", "text": "📋 Generating research methodology documentation..."}
        
        methodology_content = await self._generate_methodology(
            query=query,
            level1_queries=level1_queries,
            level2_queries=level2_queries,
            use_web_search=use_web_search
        )
        
        yield {"type": "research_summary", "content": methodology_content}
        
        yield {"type": "reasoning", "text": "✅ Research complete!"}
    
    async def _update_app_flow(
        self, 
        query: str, 
        files: Optional[List[UploadFile]], 
        existing_html: str,
        use_web_search: bool
    ):
        """
        Update existing app with or without web search
        """
        
        yield {"type": "reasoning", "text": "🎨 Updating existing app..."}
        
        new_data = None
        searched_queries = []
        tables_added = []
        
        if use_web_search:
            # Need to search for new data
            yield {"type": "reasoning", "text": "🔍 Searching for new data..."}
            
            search_info = await self.conversation._generate_search_query(query)
            
            if search_info["search_needed"] and search_info["query"]:
                yield {"type": "search_query", "text": search_info["query"]}
                searched_queries.append(search_info["query"])
                
                search_results = await self.conversation.google_search(search_info["query"])
                yield {"type": "sources", "content": search_results}
                
                browser_pool = await get_browser_pool()
                urls = [result["url"] for result in search_results[:5]]
                
                if urls:
                    yield {"type": "reasoning", "text": "📊 Extracting tables..."}
                    
                    scrape_results = await scrape_tables_parallel(
                        urls,
                        browser_pool=browser_pool,
                        timeout=60000
                    )
                    
                    tables = []
                    for url, url_tables in scrape_results.items():
                        if url_tables:
                            for table in url_tables:
                                tables.append({
                                    'url': url,
                                    'table': table
                                })
                    
                    if tables:
                        yield {"type": "tables", "content": tables}
                        tables_added = tables
                        new_data = {"tables": tables}
        
        # Update HTML (with or without new data)
        updated_html = await self._update_html_app(existing_html, query, new_data)
        
        yield {"type": "html_app", "content": updated_html}
        
        # ALWAYS Generate Update Summary
        yield {"type": "reasoning", "text": "📋 Generating update summary..."}
        
        update_summary = await self._generate_update_summary(
            modification_request=query,
            use_web_search=use_web_search,
            searched_queries=searched_queries,
            tables_added=tables_added,
            existing_html=existing_html,
            updated_html=updated_html
        )
        
        yield {"type": "research_summary", "content": update_summary}
        
        yield {"type": "reasoning", "text": "✅ App updated!"}
    
    async def _create_html_app(self, research_data: Dict) -> str:
        """
        Create NEW HTML app from research data
        Uses Opus for better instruction following
        """
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        # Build research data context
        content_by_branch = []
        for i, branch in enumerate(research_data["branches"], 1):
            branch_info = f"Branch {i}: {branch['title']}\n"
            for j, sub_query in enumerate(branch["sub_queries"], 1):
                branch_info += f"\n  Sub-query {i}.{j}: {sub_query['question']}\n"
                
                if sub_query["tables"]:
                    branch_info += f"  Tables: {sub_query['tables_count']}\n"
                    for idx, table in enumerate(sub_query["tables"][:10], 1):
                        branch_info += f"\n  Table {i}.{j}.{idx} from {table['url']}:\n"
                        branch_info += f"  {table['table']}\n"
            
            content_by_branch.append(branch_info)
        
        research_context = "\n\n".join(content_by_branch)
        
        html_prompt = f"""The current date is {current_date}.

    Create an HTML website for: "{research_data["query"]}"

    Available data:
    {research_context}

    IMPORTANT: 
    - Do EXACTLY what the user asked for, nothing more
    - If they asked for "blank with red bg", just give red background
    - If they asked for research, present the data clearly
    - Don't add unnecessary features unless requested
    - Keep it focused on their actual request
    - You must include all tables that are given in context if they are relevant to the query.
    - Add your analysis, interpretation of data and use your own trained knowledge in desgning the website.

    Technical requirements:
    - This prompt MUST always result in a new website being created - so terms like platform, app, website, portal, page ...etc all refer to a website.
    - Self-contained HTML file (inline CSS/JS)
    - Use CDN only if needed (Tailwind, Chart.js)
    - Responsive design

    Return only the HTML code."""

        # ✅ Use STREAMING for Opus (required for long operations)
        html_content = ""
        async with self.client.messages.stream(
            model=self.html_model,
            max_tokens=16000,
            messages=[{"role": "user", "content": html_prompt}]
        ) as stream:
            async for chunk in stream:
                if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        html_content += chunk.delta.text
        
        # Extract HTML if wrapped in code blocks
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0].strip()
        elif "```" in html_content:
            html_content = html_content.split("```")[1].split("```")[0].strip()
        
        return html_content

    async def _update_html_app(
        self,
        existing_html: str,
        modification_request: str,
        new_data: Optional[Dict] = None
    ) -> str:
        """
        Update EXISTING HTML app with modifications
        Uses Opus for precise updates
        """
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        new_data_context = ""
        if new_data and "tables" in new_data:
            new_data_context = "\n\nNew data available:\n"
            for idx, table in enumerate(new_data["tables"][:10], 1):
                new_data_context += f"\nTable {idx} from {table['url']}:\n{table['table']}\n"
        
        update_prompt = f"""The current date is {current_date}.

    User's update request: "{modification_request}"
    {new_data_context}

    Current HTML:
    ```html
    {existing_html}
    ```

    CRITICAL INSTRUCTIONS:
    - You must include all tables that are given in context if they are relevant to the query.
    - Add your analysis, interpretation of data and use your own trained knowledge in desgning the website.
    - Make ONLY the changes the user specifically requested
    - If they said "make background blue", ONLY change the background color
    - If they said "add data about X", ONLY add that data
    - Don't redesign or modify things they didn't mention
    - Keep all existing functionality unless asked to change it
    - Preserve the structure and style unless specifically requested to change

    Return only the complete updated HTML code."""

        # ✅ Use STREAMING for Opus
        html_content = ""
        async with self.client.messages.stream(
            model=self.html_model,
            max_tokens=16000,
            messages=[{"role": "user", "content": update_prompt}]
        ) as stream:
            async for chunk in stream:
                if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        html_content += chunk.delta.text
        
        # Extract HTML if wrapped in code blocks
        if "```html" in html_content:
            html_content = html_content.split("```html")[1].split("```")[0].strip()
        elif "```" in html_content:
            html_content = html_content.split("```")[1].split("```")[0].strip()
        
        return html_content

    async def _generate_methodology(
        self,
        query: str,
        level1_queries: List[str],
        level2_queries: Dict[str, List[str]],
        use_web_search: bool
    ) -> str:
        """
        Generate research methodology summary
        ALWAYS called for create_app flow
        """
        
        from simple_search_claude_streaming_with_web_search import ClaudeConversation
        methodology_conversation = ClaudeConversation()
        
        if use_web_search:
            # Methodology for web search research
            total_searches = sum(len(queries) for queries in level2_queries.values()) + len(level1_queries)
            total_tables = sum(len(tables) for tables in self.query_tables.values())
            
            query_breakdown = ""
            for i, l1_query in enumerate(level1_queries, 1):
                query_breakdown += f"\n**Branch {i}:** {l1_query}\n"
                l2_queries = level2_queries.get(l1_query, [])
                for j, l2_query in enumerate(l2_queries, 1):
                    tables_count = len(self.query_tables.get(l2_query, []))
                    query_breakdown += f"  - Sub-query {i}.{j}: {l2_query} ({tables_count} tables)\n"
            
            methodology_prompt = f"""Research question: "{query}"

Research structure:
- Branches: {len(level1_queries)}
- Total queries: {total_searches}
- Tables analyzed: {total_tables}

Query breakdown:
{query_breakdown}

Create a detailed research methodology summary with:

1. **Methodology Overview Table**
   - Research approach
   - Data sources (web search)
   - Queries executed
   - Tables extracted

2. **Research Process Breakdown Table**
   - Each branch and sub-queries
   - Sources consulted per query
   - Data quality assessment

3. **Data Collection Summary**
   - Web scraping methodology
   - Search strategy
   - Table extraction process

4. **Quality Assessment**
   - Source reliability
   - Data completeness
   - Limitations encountered

5. **Step-by-step Process**
   - Query decomposition
   - Systematic search execution
   - Data extraction and organization

6. **Transparency & Limitations**
   - What worked well
   - Data gaps identified
   - Recommendations for follow-up

Be detailed and academic."""
        
        else:
            # Methodology for LLM knowledge-based research
            query_breakdown = ""
            for i, l1_query in enumerate(level1_queries, 1):
                query_breakdown += f"\n**Branch {i}:** {l1_query}\n"
                l2_queries = level2_queries.get(l1_query, [])
                for j, l2_query in enumerate(l2_queries, 1):
                    query_breakdown += f"  - Sub-query {i}.{j}: {l2_query}\n"
            
            methodology_prompt = f"""Research question: "{query}"

Research structure:
- Branches: {len(level1_queries)}
- Data source: LLM training knowledge (no web search)

Query breakdown:
{query_breakdown}

Create a methodology summary explaining:

1. **Methodology Overview Table**
   - Research approach (multi-branch analysis)
   - Data source (LLM training data, knowledge cutoff)
   - Branches explored

2. **Research Process**
   - How the topic was broken down into branches
   - Sub-queries developed for each branch
   - Content generated from training knowledge

3. **Knowledge Base & Limitations**
   - Training data cutoff date
   - Scope of available knowledge
   - What types of information were included

4. **Quality Considerations**
   - Strengths of LLM knowledge approach
   - Limitations (no real-time data, no specific sources)
   - When web search would be beneficial

5. **Transparency**
   - This analysis is based on training data, not live sources
   - No tables or current statistics included
   - Conceptual and educational focus

Be clear about the methodology used."""
        
        methodology_content = ""
        async for chunk in methodology_conversation.send_message(methodology_prompt, simple_search=False):
            if chunk["type"] == "content":
                methodology_content += chunk["text"]
        
        return methodology_content
    
    async def _generate_update_summary(
        self,
        modification_request: str,
        use_web_search: bool,
        searched_queries: List[str],
        tables_added: List[Dict],
        existing_html: str,
        updated_html: str
    ) -> str:
        """
        Generate update summary explaining what was changed
        ALWAYS called for update_app flow
        """
        
        from simple_search_claude_streaming_with_web_search import ClaudeConversation
        summary_conversation = ClaudeConversation()
        
        # Build context about changes
        web_search_context = ""
        if use_web_search and searched_queries:
            web_search_context = f"""
**Web Search Performed:**
- Queries executed: {len(searched_queries)}
- Tables added: {len(tables_added)}

Search queries:
{chr(10).join(f"- {q}" for q in searched_queries)}

Tables added:
{chr(10).join(f"- Table from {t['url']}" for t in tables_added[:5])}
"""
        else:
            web_search_context = "**No web search performed** - Changes based on styling/UI modifications only"
        
        # Calculate size change
        size_before = len(existing_html)
        size_after = len(updated_html)
        size_change = size_after - size_before
        
        summary_prompt = f"""An HTML app was just updated based on user request.

**User's modification request:** "{modification_request}"

{web_search_context}

**Technical details:**
- HTML size before: {size_before} characters
- HTML size after: {size_after} characters
- Size change: {'+' if size_change > 0 else ''}{size_change} characters

Create a comprehensive update summary with:

1. **Update Summary Table**
   | Aspect | Details |
   |--------|---------|
   | Modification Type | [Styling/Data Addition/Layout/etc] |
   | Web Search Used | {use_web_search} |
   | New Data Added | {len(tables_added)} tables |
   | Code Size Change | {size_change} chars |

2. **What Was Changed**
   - Detailed list of modifications made
   - If styling: what visual changes
   - If data: what information was added
   - If layout: what structural changes

3. **Data Added** (if applicable)
   - List of new tables/information incorporated
   - Sources of new data
   - How it was integrated into the app

4. **Technical Details**
   - Code modifications made
   - New libraries/components added (if any)
   - DOM structure changes
   - CSS/JavaScript updates

5. **Impact Assessment**
   - How this improves the app
   - User experience enhancements
   - Data completeness improvements

6. **Quality & Verification**
   - Were changes successfully applied
   - Data accuracy considerations
   - Potential follow-up improvements

Be detailed about what changed and why."""

        summary_content = ""
        async for chunk in summary_conversation.send_message(summary_prompt, simple_search=False):
            if chunk["type"] == "content":
                summary_content += chunk["text"]
        
        return summary_content
    
    async def _extract_tables_from_urls(self, urls: List[str]) -> List[Dict[str, any]]:
        """Extract tables from URLs using the scraper"""
        try:
            scrape_results = await scrape_tables_parallel(
                urls,
                browser_pool=self.browser_pool,
                timeout=60000
            )
            
            all_tables = []
            for url, tables in scrape_results.items():
                if tables:
                    for table in tables:
                        all_tables.append({
                            'url': url,
                            'table': table
                        })
            
            return all_tables
        
        except Exception as e:
            print(f"❌ Table extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _generate_level1_queries(self, query: str) -> List[str]:
        """Generate 1-2 broad sub-questions"""
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        prompt = f"""The current date is {current_date}.

Given this research question: "{query}"

Break it down into 2-3 distinct, broad sub-questions that would comprehensively cover different aspects of the topic.

Rules:
- Focus ONLY on the subject matter/content, NOT on how to build, develop, create, or present it
- If the query mentions creating something (website, app, report, presentation, dashboard, tool), generate questions about the TOPIC ITSELF, not the creation process
- Each sub-question should explore a different angle or dimension of the core subject
- Questions should be broad enough to warrant multiple detailed searches
- Cover: definitions, current state, historical context, key data/metrics, trends, challenges, future outlook, comparisons, expert analysis, etc.
- Questions should be independently answerable
- Do NOT generate questions about frameworks, APIs, UI/UX, technical requirements, design patterns, or development processes

Examples:
- Query: "build a dashboard for Tesla stock"
- Good: ["What is Tesla's current stock performance, valuation metrics, and market position compared to competitors?", "What are the key factors, risks, and analyst predictions affecting Tesla stock price?", "What is Tesla's historical stock performance and how has it responded to major company events?"]
- Bad: ["What charting libraries work best for stock dashboards?", "How to design a responsive stock tracking UI?"]

- Query: "create a report on climate change impacts"
- Good: ["What are the current observed effects of climate change on global ecosystems and weather patterns?", "What are the projected economic and social impacts of climate change through 2050?", "What mitigation strategies are proving most effective and what challenges remain?"]
- Bad: ["What report templates work best for climate data?", "How to visualize climate statistics effectively?"]

Format your response ONLY as a JSON array of strings:
["question 1", "question 2", "question 3"]"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            queries = json.loads(response_text)
            return queries[:1]
        except:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
            return queries
    
    async def _generate_level2_queries(self, l1_query: str, original_query: str) -> List[str]:
        """Generate 1-2 specific questions for each L1 query"""
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        prompt = f"""The current date is {current_date}.

    Original research question: "{original_query}"

    Sub-question to expand: "{l1_query}"

    Generate 1-2 specific, searchable queries that would help answer this sub-question in detail.

    Rules:
    - Focus ONLY on the subject matter/content, NOT on how to build, develop, create, or present it
    - If the original query mentions creating something (website, app, report, presentation), generate queries about the TOPIC ITSELF, not the creation process
    - Each query should be specific and directly searchable on Google
    - Queries should dig deeper into the sub-question's core subject
    - Include current year (2025) if time-sensitive
    - Make queries independently understandable (include context)
    - Do NOT generate queries about frameworks, APIs, UI/UX, technical requirements, or development processes

    Example:
    - Original: "develop a website for MSFT stock"
    - Sub-question: "What is Microsoft's current stock performance?"
    - Good: ["MSFT stock price performance Q4 2025", "Microsoft revenue earnings report 2025"]
    - Bad: ["best charts library for stock visualization", "real-time stock API providers"]

    Format your response ONLY as a JSON array of strings:
    ["specific query 1", "specific query 2"]"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            queries = json.loads(response_text)
            return queries
        except:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
            return queries