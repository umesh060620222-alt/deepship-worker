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


async def classify_web_search_needed(query: str, client, model: str) -> bool:
    """
    Simple classifier for first message: Does this need web search?
    Returns: True if web search needed, False if LLM knowledge sufficient
    """
    
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    prompt = f"""The current date is {current_date}.

User query: "{query}"

Does this query require WEB SEARCH for current/factual data, or can it be answered with LLM knowledge alone?

Web search NEEDED for:
- Current data, statistics, prices, news
- Comparative analysis with real numbers (e.g., "Apple vs Microsoft revenue")
- Time-sensitive information (2024, 2025, "latest", "current")
- Factual research requiring sources

LLM knowledge SUFFICIENT for:
- General knowledge topics (e.g., "types of cheese", "history of Rome")
- Creative/educational content (e.g., "build a quiz", "create a calculator")
- Conceptual explanations
- Historical facts (pre-2024)

Return ONLY valid JSON:
{{"use_web_search": true}} OR {{"use_web_search": false}}"""

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
        use_search = result.get("use_web_search", True)
        
        print(f"🔍 Web search needed: {use_search}")
        return use_search
    
    except Exception as e:
        print(f"⚠️ Classifier error: {e}, defaulting to True (use web search)")
        return True


async def classify_followup_message(
    conversation_messages: List[Dict], 
    user_message: str, 
    client, 
    model: str,
    has_existing_md: bool
) -> Dict[str, any]:
    """
    Full classifier for follow-up messages
    Returns: {"route": "conversation|create_report|update_report", "use_web_search": bool}
    """
    
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # ✅ Format conversation history safely
    history_text = ""
    for msg in conversation_messages[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Handle different content types
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        text_parts.append("[Image]")
                    elif block.get("type") == "document":
                        text_parts.append("[Document]")
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
        
        if content:
            history_text += f"{role}: {content[:500]}...\n\n"
    
    existing_md_context = ""
    if has_existing_md:
        existing_md_context = "\n\nNOTE: There is an existing markdown report that can be updated."
    
    router_prompt = f"""The current date is {current_date}.

Analyze the conversation and classify the user's message.

Recent conversation history:
{history_text}{existing_md_context}

Latest user message: "{user_message}"

Classify the message:

1. ROUTE (choose one):

   a) "conversation" - Just answering questions, NO report creation/update
      - Asking about existing content
      - Clarifying questions
      - Discussion
   
   b) "create_report" - Create NEW report (topic is significantly DIFFERENT from current conversation)
      - New topic unrelated to current conversation
      - User wants to research something completely different
      - Examples: Currently discussing "Apple" → User asks "Research climate change"
   
   c) "update_report" - Update EXISTING report (topic is RELATED to current conversation)
      - Extending/modifying current topic
      - Adding data to existing research
      - Content changes
      - Examples: Currently discussing "Apple vs Microsoft" → User asks "Add revenue data"
      - ONLY if existing markdown report is present

2. USE_WEB_SEARCH (true/false):
   
   true = Need current/factual data from web
   - Current statistics, prices, news
   - Comparative analysis with real numbers
   - Time-sensitive info ("2025", "latest", "current")
   
   false = LLM knowledge sufficient
   - General knowledge topics
   - Creative/educational content
   - Conceptual explanations

CRITICAL RULES:
- If no existing markdown report → cannot choose "update_report"
- "conversation" never needs web search (always false)
- New unrelated topic → "create_report"
- Related/extending topic + existing report → "update_report"

Return ONLY valid JSON:
{{"route": "conversation|create_report|update_report", "use_web_search": true|false, "reasoning": "..."}}"""

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
        
        # Fallback: if update_report but no existing markdown, change to create_report
        if route == "update_report" and not has_existing_md:
            print(f"⚠️ Router chose 'update_report' but no existing markdown, changing to 'create_report'")
            route = "create_report"
        
        # Conversation never uses web search
        if route == "conversation":
            use_web_search = False
        
        print(f"🔀 Router: {route}, Web search: {use_web_search} - {result.get('reasoning', '')}")
        return {"route": route, "use_web_search": use_web_search}
    
    except Exception as e:
        print(f"⚠️ Classifier error: {e}, defaulting to conversation")
        return {"route": "conversation", "use_web_search": False}


class MarkdownResearch:
    """
    Markdown-based research system with multi-stage deep research and smart routing
    """
    
    def __init__(self, conversation):
        """
        Args:
            conversation: ClaudeConversation instance
        """
        self.conversation = conversation
        self.client = conversation.client
        self.model = conversation.model  # Sonnet for queries/routing
        self.report_model = "claude-opus-4-20250514"  # Opus for markdown generation
        self.browser_pool = None
        self.query_tables = {}
        self.last_research_data = None
    
    async def research(
        self, 
        query: str, 
        files: Optional[List[UploadFile]] = None,
        existing_markdown: Optional[str] = None
    ):
        """
        Main entry point with intelligent routing
        
        Args:
            query: User's message/question
            files: Optional uploaded files
            existing_markdown: Optional existing markdown report to update
        
        Yields: {"type": "search_query"/"sources"/"tables"/"reasoning"/"content"/"markdown_report"/"research_summary"}
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
                # Create report
                async for chunk in self._create_report_flow(query, files, decision["use_web_search"]):
                    yield chunk
        
        else:
            # ✅ FOLLOW-UP MESSAGE - Classify route + web search
            decision = await classify_followup_message(
                self.conversation.messages,
                query,
                self.client,
                self.model,
                has_existing_md=existing_markdown is not None
            )
            
            route = decision["route"]
            use_web_search = decision["use_web_search"]
            
            if route == "conversation":
                # Just answer, no report
                yield {"type": "reasoning", "text": "💬 Answering your question..."}
                async for chunk in self.conversation.send_message(query, files, simple_search=False):
                    yield chunk
            
            elif route == "create_report":
                # New topic - create new report
                async for chunk in self._create_report_flow(query, files, use_web_search):
                    yield chunk
            
            elif route == "update_report":
                # Update existing report
                async for chunk in self._update_report_flow(query, files, existing_markdown, use_web_search):
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
                max_tokens=100,
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
    
    async def _create_report_flow(self, query: str, files: Optional[List[UploadFile]], use_web_search: bool):
        """
        Create new markdown report with multi-stage deep research
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
        
        # Phase 4: Generate markdown report
        yield {"type": "reasoning", "text": "📝 Generating markdown research report..."}
        
        markdown_report = await self._create_markdown_report(research_data)
        
        yield {"type": "markdown_report", "content": markdown_report}
        
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
    
    async def _update_report_flow(
        self, 
        query: str, 
        files: Optional[List[UploadFile]], 
        existing_markdown: str,
        use_web_search: bool
    ):
        """
        Update existing markdown report with or without web search
        """
        
        yield {"type": "reasoning", "text": "📝 Updating existing report..."}
        
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
        
        # Update markdown (with or without new data)
        updated_markdown = await self._update_markdown_report(existing_markdown, query, new_data)
        
        yield {"type": "markdown_report", "content": updated_markdown}
        
        # ALWAYS Generate Update Summary
        yield {"type": "reasoning", "text": "📋 Generating update summary..."}
        
        update_summary = await self._generate_update_summary(
            modification_request=query,
            use_web_search=use_web_search,
            searched_queries=searched_queries,
            tables_added=tables_added,
            existing_markdown=existing_markdown,
            updated_markdown=updated_markdown
        )
        
        yield {"type": "research_summary", "content": update_summary}
        
        yield {"type": "reasoning", "text": "✅ Report updated!"}
    
    async def _create_markdown_report(self, research_data: Dict) -> str:
        """
        Create comprehensive markdown report from research data
        Generates report for each query + final synthesis
        """
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        # Generate individual query reports
        all_query_reports = []
        
        for i, branch in enumerate(research_data["branches"], 1):
            for j, sub_query in enumerate(branch["sub_queries"], 1):
                tables_context = ""
                if sub_query.get('tables'):
                    tables_context = "\n\nAvailable data:\n" + "\n\n".join([
                        f"Table from {t['url']}:\n{t['table']}" 
                        for t in sub_query['tables']
                    ])
                
                query_prompt = f"""The current date is {current_date}.

Query: "{sub_query['question']}"
{tables_context}

Write a focused and detailed report section for this query. Use your own self knowledge to write a detailed report that is full with details, interpretations and analysis.
You MUST return ALL tables relevant to the Query in the response. You MUST create any additional tables that are relevant for the Query.

Return only markdown."""

                # Generate report for this query
                query_report = ""
                async with self.client.messages.stream(
                    model=self.report_model,  # Use Opus for quality
                    max_tokens=4000,
                    messages=[{"role": "user", "content": query_prompt}]
                ) as stream:
                    async for chunk in stream:
                        if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                query_report += chunk.delta.text
                
                all_query_reports.append({
                    'branch': branch['title'],
                    'query': sub_query['question'],
                    'report': query_report
                })
        
        # Generate final synthesis
        synthesis_context = "\n\n".join([
            f"## {r['query']}\n{r['report']}" 
            for r in all_query_reports
        ])
        
        synthesis_prompt = f"""The current date is {current_date}.

Research question: "{research_data['query']}"

Individual query reports:
{synthesis_context}

Create a final synthesis that summarizes key findings and provides conclusions. Use the reports provided.

Return only markdown."""

        final_synthesis = ""
        async with self.client.messages.stream(
            model=self.report_model,
            max_tokens=8000,
            messages=[{"role": "user", "content": synthesis_prompt}]
        ) as stream:
            async for chunk in stream:
                if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        final_synthesis += chunk.delta.text
        
        # Assemble complete markdown report
        markdown_report = f"""# {research_data['query']}

*Generated on {current_date}*

---

"""
        
        # Append each query report
        for r in all_query_reports:
            markdown_report += f"""## {r['query']}

{r['report']}

---

"""
        
        # Append final synthesis
        markdown_report += f"""# Final Synthesis

{final_synthesis}
"""
        
        return markdown_report
    
    async def _update_markdown_report(
        self,
        existing_markdown: str,
        modification_request: str,
        new_data: Optional[Dict] = None
    ) -> str:
        """
        Update EXISTING markdown report with modifications
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

Current markdown report:
```markdown
{existing_markdown}
```

Make only the changes requested. Use the data provided if applicable. Preserve everything else.

Return only the complete updated markdown."""

        # ✅ Use STREAMING for Opus
        markdown_content = ""
        async with self.client.messages.stream(
            model=self.report_model,
            max_tokens=16000,
            messages=[{"role": "user", "content": update_prompt}]
        ) as stream:
            async for chunk in stream:
                if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        markdown_content += chunk.delta.text
        
        # Extract markdown if wrapped in code blocks
        if "```markdown" in markdown_content:
            markdown_content = markdown_content.split("```markdown")[1].split("```")[0].strip()
        elif "```" in markdown_content:
            markdown_content = markdown_content.split("```")[1].split("```")[0].strip()
        
        return markdown_content
    
    async def _generate_methodology(
        self,
        query: str,
        level1_queries: List[str],
        level2_queries: Dict[str, List[str]],
        use_web_search: bool
    ) -> str:
        """
        Generate research methodology summary
        ALWAYS called for create_report flow
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
        existing_markdown: str,
        updated_markdown: str
    ) -> str:
        """
        Generate update summary explaining what was changed
        ALWAYS called for update_report flow
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
            web_search_context = "**No web search performed** - Changes based on content modifications only"
        
        # Calculate size change
        size_before = len(existing_markdown)
        size_after = len(updated_markdown)
        size_change = size_after - size_before
        
        summary_prompt = f"""A markdown report was just updated based on user request.

**User's modification request:** "{modification_request}"

{web_search_context}

**Technical details:**
- Markdown size before: {size_before} characters
- Markdown size after: {size_after} characters
- Size change: {'+' if size_change > 0 else ''}{size_change} characters

Create a comprehensive update summary with:

1. **Update Summary Table**
   | Aspect | Details |
   |--------|---------|
   | Modification Type | [Content Addition/Data Update/Restructure/etc] |
   | Web Search Used | {use_web_search} |
   | New Data Added | {len(tables_added)} tables |
   | Content Size Change | {size_change} chars |

2. **What Was Changed**
   - Detailed list of modifications made
   - If content: what sections were added/modified
   - If data: what information was added
   - If structure: what organizational changes

3. **Data Added** (if applicable)
   - List of new tables/information incorporated
   - Sources of new data
   - How it was integrated into the report

4. **Technical Details**
   - Content modifications made
   - New sections added
   - Structural changes
   - Data formatting updates

5. **Impact Assessment**
   - How this improves the report
   - Content completeness improvements
   - Research depth enhancements

6. **Quality & Verification**
   - Were changes successfully applied
   - Data accuracy considerations
   - Potential follow-up improvements

Be detailed about what changed and why."""

        summary_content = ""
        
        try:
            async for chunk in summary_conversation.send_message(summary_prompt, simple_search=False):
                if chunk["type"] == "content":
                    summary_content += chunk["text"]
            
            return summary_content
        except Exception as e:
            print("Exception in summary development " + str(e))
            
            return "Task Completed."
        
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
        """Generate 2-3 broad sub-questions"""
        
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
            print("LEVEL 1 Queries : " + str(queries))
            return queries
        
        except:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
            return queries
    
#     async def _generate_level2_queries(self, l1_query: str, original_query: str) -> List[str]:
#         """Generate 1-2 specific questions for each L1 query"""
        
#         current_date = datetime.now().strftime("%A, %B %d, %Y")
        
#         prompt = f"""The current date is {current_date}.

# Original research question: "{original_query}"

# Sub-question to expand: "{l1_query}"

# Generate 1-2 specific, searchable queries that would help answer this sub-question in detail.

# Rules:
# - Each query should be specific and directly searchable on Google
# - Queries should dig deeper into the sub-question
# - Include current year (2025) if time-sensitive
# - Make queries independently understandable (include context)

# Format your response ONLY as a JSON array of strings:
# ["specific query 1", "specific query 2"]"""

#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=500,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         response_text = response.content[0].text.strip()
        
#         try:
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             queries = json.loads(response_text)
#             return queries[:2]
#         except:
#             lines = [line.strip() for line in response_text.split('\n') if line.strip()]
#             queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
#             return queries[:2]

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
            print("LEVEL 2 Queries : " + str(queries))
            return queries
        except:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
            return queries













# import anthropic
# import json
# from typing import List, Dict, Optional
# from fastapi import UploadFile
# import asyncio
# from tables_scraper import scrape_tables_parallel, BrowserPool

# # ✅ Global browser pool (shared across requests)
# _global_browser_pool: Optional[BrowserPool] = None
# _pool_lock = asyncio.Lock()

# async def get_browser_pool():
#     """Get or create global browser pool"""
#     global _global_browser_pool
    
#     async with _pool_lock:
#         if _global_browser_pool is None or not _global_browser_pool.initialized:
#             _global_browser_pool = BrowserPool(pool_size=2, max_tabs_per_browser=10)
#             await _global_browser_pool.initialize()
    
#     return _global_browser_pool


# async def classify_message(conversation_messages: List[Dict], user_message: str, client, model: str) -> str:
#     """
#     Router: Classify message routing
#     Returns: "deep_research" | "web_search" | "conversation"
#     """
    
#     from datetime import datetime
#     current_date = datetime.now().strftime("%A, %B %d, %Y")
    
#     # Format conversation history
#     history_text = ""
#     for msg in conversation_messages[-6:]:  # Last 6 messages for context
#         role = msg.get("role", "")
#         content = msg.get("content", "")
#         if isinstance(content, list):
#             content = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
#         history_text += f"{role}: {content[:500]}...\n\n"  # More context per message
    
#     router_prompt = f"""You are a routing classifier. Analyze this conversation and determine what action to take for the latest user message.

# Recent conversation history:
# {history_text}

# Latest user message: "{user_message}"

# Classify as:

# 1. "deep_research" - User wants comprehensive research on a NEW topic requiring multiple queries, branches, and deep analysis
#    - Indicators: New broad topic, "research X", "analyze comprehensively", requires multiple angles
   
# 2. "web_search" - User needs NEW information that requires web search
#    - Indicators: Asking for data NOT in the conversation above, current events, specific facts not yet covered
   
# 3. "conversation" - User is asking about information ALREADY in the conversation above
#    - Indicators: "What did you say about...", asking to clarify/explain existing data, referencing previous findings

# CRITICAL: Look carefully at the conversation history. If the user is asking about data that was ALREADY provided above, choose "conversation", NOT "web_search".

# Return ONLY valid JSON:
# {{"route": "deep_research|web_search|conversation", "reasoning": "..."}}"""

#     try:
#         response = await client.messages.create(
#             model=model,
#             max_tokens=200,
#             system_message = f"The current date is {current_date}."
#             messages=[{"role": "user", "content": router_prompt}],
#             existing_html = None
#         )
        
#         response_text = response.content[0].text.strip()
        
#         # Parse JSON
#         if "```json" in response_text:
#             response_text = response_text.split("```json")[1].split("```")[0].strip()
#         elif "```" in response_text:
#             response_text = response_text.split("```")[1].split("```")[0].strip()
        
#         result = json.loads(response_text)
#         route = result.get("route", "conversation")
        
#         print(f"🔀 Router decision: {route} - {result.get('reasoning', '')}")
#         return route
    
#     except Exception as e:
#         print(f"⚠️ Router error: {e}, defaulting to 'conversation'")
#         return "conversation"
    

# class DeepResearch:
#     """
#     Tree-based research: Input → Level 1 (1-2 queries) → Level 2 (1-2 queries each)
#     Calls send_message for each query with table context
#     """
    
#     def __init__(self, conversation):
#         """
#         Args:
#             conversation: ClaudeConversation instance
#         """
#         self.conversation = conversation
#         self.client = conversation.client
#         self.model = conversation.model
#         self.browser_pool = None
#         self.query_tables = {}  # Store tables per query
#         self.query_results = []  # Store all query results
    
    
#     async def research(self, query: str, files: Optional[List[UploadFile]] = None):
#         """
#         Main entry point with automatic routing logic
        
#         Args:
#             query: User's research question
#             files: Optional uploaded files
        
#         Yields: {"type": "search_query"/"sources"/"tables"/"reasoning"/"content"/"report"/"research_summary"}
#         """
        
#         # ✅ Automatically detect if this is a follow-up
#         follow_up = len(self.conversation.messages) > 0
        
#         # ✅ ROUTING LOGIC
#         if follow_up:
#             # Call router to decide: deep_research | web_search | conversation
#             route = await classify_message(
#                 self.conversation.messages,
#                 query,
#                 self.client,
#                 self.model
#             )
            
#             if route == "conversation":
#                 # Answer from existing context, NO web search
#                 yield {"type": "reasoning", "text": "💬 Answering from existing context..."}
#                 async for chunk in self.conversation.send_message(query, files,simple_search = False):
#                     yield chunk
#                 return  # Exit after conversation
            
#             elif route == "web_search":
#                 # Need to search for NEW information
#                 yield {"type": "reasoning", "text": "🔍 Searching for new information..."}
                
#                 # Check if web search is needed
#                 search_info = await self.conversation._generate_search_query(query)
                
#                 if search_info["search_needed"] and search_info["query"]:
#                     yield {"type": "search_query", "text": search_info["query"]}
                    
#                     # Perform web search
#                     search_results = await self.conversation.google_search(search_info["query"])
#                     yield {"type": "sources", "content": search_results}
                    
#                     # Extract tables
#                     browser_pool = await get_browser_pool()
#                     urls = [result["url"] for result in search_results[:5]]
                    
#                     if urls:
#                         yield {"type": "reasoning", "text": "📊 Extracting tables from sources..."}
                        
#                         scrape_results = await scrape_tables_parallel(
#                             urls,
#                             browser_pool=browser_pool,
#                             timeout=60000
#                         )
                        
#                         # Format tables
#                         tables = []
#                         for url, url_tables in scrape_results.items():
#                             if url_tables:
#                                 for table in url_tables:
#                                     tables.append({
#                                         'url': url,
#                                         'table': table
#                                     })
                        
#                         if tables:
#                             yield {"type": "tables", "content": tables}
#                             yield {"type": "reasoning", "text": f"✅ Extracted {len(tables)} tables"}
                            
#                             # Build message with tables
#                             tables_text = "\n\n---\n\n".join([
#                                 f"**Table from {t['url']}:**\n\n{t['table']}"
#                                 for t in tables[:10]
#                             ])
                            
#                             enhanced_query = f"""{query}

#     ## Reference Tables:

#     {tables_text}

#     ## Instructions:
#     Analyze the tables and data above. If relevant, include tables in your response with your analysis."""
#                         else:
#                             enhanced_query = query
#                     else:
#                         enhanced_query = query
#                 else:
#                     enhanced_query = query
                
#                 # Stream response
#                 async for chunk in self.conversation.send_message(enhanced_query, files,simple_search = False):
#                     yield chunk
                
#                 return  # Exit after web search
            
#             # If route == "deep_research", continue to full deep research below
        
#         # ✅ DEEP RESEARCH (either initial query or router decided deep_research)
#         yield {"type": "reasoning", "text": "🔍 Starting deep research process..."}
        
#         # Use global shared browser pool
#         self.browser_pool = await get_browser_pool()
        
#         # Reset query results for new deep research
#         self.query_results = []
#         self.query_tables = {}
        
#         # Phase 1: Generate Level 1 queries (1-2 sub-questions)
#         yield {"type": "reasoning", "text": "📊 Analyzing research question and generating main branches..."}
#         level1_queries = await self._generate_level1_queries(query)
#         yield {"type": "reasoning", "text": f"✅ Generated {len(level1_queries)} research branches"}
        
#         yield {"type": "reasoning", "text": "🔎 Beginning web searches for Level 1 queries..."}
        
#         for i, q in enumerate(level1_queries, 1):
#             search_info = await self.conversation._generate_search_query(q)
#             if search_info["search_needed"] and search_info["query"]:
#                 yield {"type": "search_query", "text": search_info["query"]}
                
#                 search_results = await self.conversation.google_search(search_info["query"])
#                 yield {"type": "sources", "content": search_results}
                
#                 yield {"type": "reasoning", "text": f"📄 Extracting tables from Branch {i} sources..."}
                
#                 urls = [result["url"] for result in search_results[:5]]
#                 if urls:
#                     tables = await self._extract_tables_from_urls(urls)
#                     if tables:
#                         self.query_tables[q] = tables
#                         yield {"type": "tables", "content": tables}
#                         yield {"type": "reasoning", "text": f"✅ Extracted {len(tables)} tables from Branch {i}"}
        
#         yield {"type": "reasoning", "text": "🌳 Expanding branches into detailed sub-queries..."}
        
#         # Phase 2: Generate Level 2 queries (1-2 per Level 1)
#         level2_queries = {}
#         for i, l1_query in enumerate(level1_queries, 1):
#             l2_queries = await self._generate_level2_queries(l1_query, query)
#             level2_queries[l1_query] = l2_queries
            
#             yield {"type": "reasoning", "text": f"✅ Branch {i} expanded into {len(l2_queries)} sub-queries"}
            
#             for j, q in enumerate(l2_queries, 1):
#                 search_info = await self.conversation._generate_search_query(q)
#                 if search_info["search_needed"] and search_info["query"]:
#                     yield {"type": "search_query", "text": search_info["query"]}
                    
#                     search_results = await self.conversation.google_search(search_info["query"])
#                     yield {"type": "sources", "content": search_results}
                    
#                     urls = [result["url"] for result in search_results[:5]]
#                     if urls:
#                         tables = await self._extract_tables_from_urls(urls)
#                         if tables:
#                             self.query_tables[q] = tables
#                             yield {"type": "tables", "content": tables}
        
#         total_queries = sum(len(queries) for queries in level2_queries.values())
#         yield {"type": "reasoning", "text": f"✅ Data collection complete: {total_queries} queries executed"}
        
#         yield {"type": "reasoning", "text": "🤖 Analyzing each query with AI..."}
        
#         # Phase 3: Execute send_message for each Level 2 query with tables
#         query_counter = 0
#         for branch_num, (l1_query, l2_queries) in enumerate(level2_queries.items(), 1):
#             for query_num, l2_query in enumerate(l2_queries, 1):
#                 query_counter += 1
                
#                 yield {"type": "reasoning", "text": f"🔬 Analyzing query {query_counter}/{total_queries}..."}
                
#                 tables = self.query_tables.get(l2_query, [])
                
#                 if tables:
#                     tables_text = "\n\n---\n\n".join([
#                         f"**Table from {t['url']}:**\n\n{t['table']}"
#                         for t in tables[:10]
#                     ])
#                     query_with_tables = f"""{l2_query}

#     ## Reference Tables:

#     {tables_text}

#     ## Instructions:
#     **CRITICAL:** You MUST include relevant tables in your response.

#     1. Review all the tables above
#     2. **Copy the relevant tables directly into your response** (in markdown format)
#     3. For each table you include:
#     - Add a brief title/context
#     - Present the table
#     - Provide your analysis of what the data shows
    
#     4. Focus your analysis on:
#     - Key findings from the numerical data
#     - Patterns and trends in the numbers
#     - Notable statistics or metrics
#     - Comparisons across sources
#     - What the data reveals about the question

#     5. If a table is not relevant, skip it

#     **FORMAT:** 
#     For each relevant table, use this structure:
#     - Brief context/title
#     - The table itself (copy the markdown)
#     - Your interpretation (2-3 sentences)

#     Be analytical and data-driven. Include ALL relevant numerical data by copying the tables."""
#                 else:
#                     query_with_tables = l2_query
                
#                 result_content = ""
#                 async for chunk in self.conversation.send_message(query_with_tables, files,simple_search = False):
#                     if chunk["type"] == "content":
#                         result_content += chunk["text"]
#                         yield {"type": "content", "text": chunk["text"]}
                
#                 self.query_results.append({
#                     "branch": l1_query,
#                     "query": l2_query,
#                     "result": result_content,
#                     "tables_count": len(tables)
#                 })
        
#         yield {"type": "reasoning", "text": f"✅ Completed AI analysis for all {total_queries} queries"}
        
#         yield {"type": "reasoning", "text": "📝 Compiling comprehensive research report..."}
        
#         # Phase 4: Build final report from all query results
#         final_report = f"""# Deep Research Report

#     ## Research Question: {query}

#     ---

#     """
        
#         for i, l1_query in enumerate(level1_queries, 1):
#             final_report += f"\n## Branch {i}: {l1_query}\n\n"
            
#             branch_results = [r for r in self.query_results if r["branch"] == l1_query]
            
#             for j, result in enumerate(branch_results, 1):
#                 final_report += f"### Sub-query {i}.{j}: {result['query']}\n\n"
#                 final_report += result['result']
#                 final_report += "\n\n---\n\n"
        
#         yield {"type": "reasoning", "text": "🎯 Generating cross-branch synthesis..."}
        
#         # Phase 5: Generate final synthesis
#         synthesis_prompt = f"""Based on all the research conducted above in this conversation, create a comprehensive synthesis.

#     Original question: "{query}"

#     Review all the branch analyses above (which already include relevant tables and data) and create:

#     ### 1. Executive Summary
#     Synthesize the most important findings across all branches (3-4 paragraphs)

#     ### 2. Cross-Branch Synthesis Tables

#     Create NEW tables that consolidate findings:

#     **Key Findings Comparison Table:**
#     | Branch | Main Findings | Supporting Data | Implications |
#     |--------|---------------|-----------------|--------------|

#     **Quantitative Summary Table:**
#     | Metric/Data Point | Value | Source | Significance |
#     |-------------------|-------|--------|--------------|

#     ### 3. Patterns & Insights
#     - Cross-cutting themes
#     - Contradictions or gaps
#     - Emerging trends

#     ### 4. Overall Implications
#     What do these findings mean for the original question? What are the practical implications?

#     ### 5. Future Research Needs
#     What gaps exist? What should be investigated further?

#     Be comprehensive and analytical. Create synthesis tables that combine data from multiple branches."""

#         synthesis_content = ""
#         async for chunk in self.conversation.send_message(synthesis_prompt, files, simple_search = False):
#             if chunk["type"] == "content":
#                 synthesis_content += chunk["text"]
        
#         final_report += f"\n\n# Final Synthesis\n\n{synthesis_content}"
        
#         yield {"type": "report", "content": final_report}
        
#         yield {"type": "reasoning", "text": "📋 Generating research methodology documentation..."}
        
#         # Phase 6: Research Summary & Methodology
#         from simple_search_claude_streaming_with_web_search import ClaudeConversation
#         methodology_conversation = ClaudeConversation()
        
#         total_searches = sum(len(queries) for queries in level2_queries.values()) + len(level1_queries)
#         total_tables = sum(len(tables) for tables in self.query_tables.values())
        
#         query_breakdown = ""
#         for i, l1_query in enumerate(level1_queries, 1):
#             query_breakdown += f"\n**Branch {i}:** {l1_query}\n"
#             l2_queries = level2_queries.get(l1_query, [])
#             for j, l2_query in enumerate(l2_queries, 1):
#                 tables_count = len(self.query_tables.get(l2_query, []))
#                 query_breakdown += f"  - Sub-query {i}.{j}: {l2_query} ({tables_count} tables)\n"
        
#         methodology_prompt = f"""Research question: "{query}"

#     Research structure:
#     - Branches: {len(level1_queries)}
#     - Total queries: {total_searches}
#     - Tables analyzed: {total_tables}

#     Query breakdown:
#     {query_breakdown}

#     Create a detailed research methodology summary with:

#     1. **Methodology Overview Table**
#     2. **Research Process Breakdown Table**
#     3. **Data Collection Summary**
#     4. **Quality Assessment**
#     5. **Step-by-step Process**
#     6. **Transparency & Limitations**

#     Be detailed and academic."""

#         methodology_content = ""
#         async for chunk in methodology_conversation.send_message(methodology_prompt, files, simple_search = False):
#             if chunk["type"] == "content":
#                 methodology_content += chunk["text"]
        
#         yield {"type": "research_summary", "content": methodology_content}
        
#         yield {"type": "reasoning", "text": "✅ Deep research complete!"}
        
#     async def _extract_tables_from_urls(self, urls: List[str]) -> List[Dict[str, any]]:
#         """Extract tables from URLs using the scraper"""
#         try:
#             scrape_results = await scrape_tables_parallel(
#                 urls,
#                 browser_pool=self.browser_pool,
#                 timeout=60000
#             )
            
#             all_tables = []
#             for url, tables in scrape_results.items():
#                 if tables:
#                     for table in tables:
#                         all_tables.append({
#                             'url': url,
#                             'table': table
#                         })
            
#             return all_tables
        
#         except Exception as e:
#             print(f"❌ Table extraction error: {e}")
#             import traceback
#             traceback.print_exc()
#             return []
    
#     async def _generate_level1_queries(self, query: str) -> List[str]:
#         """Generate 1-2 broad sub-questions"""
        
#         prompt = f"""Given this research question: "{query}"

# Break it down into 1-2 distinct, broad sub-questions that would comprehensively cover different aspects of the topic.

# Rules:
# - Each sub-question should explore a different angle or dimension
# - Questions should be broad enough to warrant multiple detailed searches
# - Cover: definitions, current state, applications, challenges, future directions, comparisons, etc.
# - Questions should be independently answerable

# Format your response ONLY as a JSON array of strings:
# ["question 1", "question 2"]"""

#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=1000,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         response_text = response.content[0].text.strip()
        
#         try:
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             queries = json.loads(response_text)
#             return queries[:1]
#         except:
#             lines = [line.strip() for line in response_text.split('\n') if line.strip()]
#             queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
#             return queries[:1]
    
#     async def _generate_level2_queries(self, l1_query: str, original_query: str) -> List[str]:
#         """Generate 1-2 specific questions for each L1 query"""
        
#         prompt = f"""Original research question: "{original_query}"

# Sub-question to expand: "{l1_query}"

# Generate 1-2 specific, searchable queries that would help answer this sub-question in detail.

# Rules:
# - Each query should be specific and directly searchable on Google
# - Queries should dig deeper into the sub-question
# - Include current year (2025) if time-sensitive
# - Make queries independently understandable (include context)

# Format your response ONLY as a JSON array of strings:
# ["specific query 1", "specific query 2"]"""

#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=500,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         response_text = response.content[0].text.strip()
        
#         try:
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             queries = json.loads(response_text)
#             return queries[:1]
#         except:
#             lines = [line.strip() for line in response_text.split('\n') if line.strip()]
#             queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
#             return queries[:1]


































# import anthropic
# import json
# from typing import List, Dict, Optional
# from fastapi import UploadFile
# import asyncio
# from tables_scraper import scrape_tables_parallel, BrowserPool

# # ✅ Global browser pool (shared across requests)
# _global_browser_pool: Optional[BrowserPool] = None
# _pool_lock = asyncio.Lock()

# async def get_browser_pool():
#     """Get or create global browser pool"""
#     global _global_browser_pool
    
#     async with _pool_lock:
#         if _global_browser_pool is None or not _global_browser_pool.initialized:
#             _global_browser_pool = BrowserPool(pool_size=2, max_tabs_per_browser=10)
#             await _global_browser_pool.initialize()
    
#     return _global_browser_pool

# class DeepResearch:
#     """
#     Tree-based research: Input → Level 1 (1-2 queries) → Level 2 (1-2 queries each)
#     Calls send_message for each query with table context
#     """
    
#     def __init__(self, conversation):
#         """
#         Args:
#             conversation: ClaudeConversation instance
#         """
#         self.conversation = conversation
#         self.client = conversation.client
#         self.model = conversation.model
#         self.browser_pool = None
#         self.query_tables = {}  # Store tables per query
#         self.query_results = []  # Store all query results
    
#     async def research(self, query: str, files: Optional[List[UploadFile]] = None):
#         """
#         Main entry point - same signature as send_message()
#         Yields: {"type": "search_query"/"sources"/"tables"/"reasoning"/"content"/"report"/"research_summary"}
#         """
        
#         # ✅ Use global shared browser pool
#         self.browser_pool = await get_browser_pool()
        
#         # Milestone 1
#         yield {"type": "reasoning", "text": "🔍 Starting deep research process..."}
        
#         # Phase 1: Generate Level 1 queries (1-2 sub-questions)
#         yield {"type": "reasoning", "text": "📊 Analyzing research question and generating main branches..."}
#         level1_queries = await self._generate_level1_queries(query)
#         yield {"type": "reasoning", "text": f"✅ Generated {len(level1_queries)} research branches"}
        
#         # Milestone 2
#         yield {"type": "reasoning", "text": "🔎 Beginning web searches for Level 1 queries..."}
        
#         for i, q in enumerate(level1_queries, 1):
#             search_info = await self.conversation._generate_search_query(q)
#             if search_info["search_needed"] and search_info["query"]:
#                 yield {"type": "search_query", "text": search_info["query"]}
                
#                 search_results = await self.conversation.google_search(search_info["query"])
#                 yield {"type": "sources", "content": search_results}
                
#                 # Milestone 3
#                 yield {"type": "reasoning", "text": f"📄 Extracting tables from Branch {i} sources..."}
                
#                 urls = [result["url"] for result in search_results[:5]]
#                 if urls:
#                     tables = await self._extract_tables_from_urls(urls)
#                     if tables:
#                         self.query_tables[q] = tables
#                         yield {"type": "tables", "content": tables}
#                         yield {"type": "reasoning", "text": f"✅ Extracted {len(tables)} tables from Branch {i}"}
        
#         # Milestone 4
#         yield {"type": "reasoning", "text": "🌳 Expanding branches into detailed sub-queries..."}
        
#         # Phase 2: Generate Level 2 queries (1-2 per Level 1)
#         level2_queries = {}
#         for i, l1_query in enumerate(level1_queries, 1):
#             l2_queries = await self._generate_level2_queries(l1_query, query)
#             level2_queries[l1_query] = l2_queries
            
#             # Milestone 5
#             yield {"type": "reasoning", "text": f"✅ Branch {i} expanded into {len(l2_queries)} sub-queries"}
            
#             for j, q in enumerate(l2_queries, 1):
#                 search_info = await self.conversation._generate_search_query(q)
#                 if search_info["search_needed"] and search_info["query"]:
#                     yield {"type": "search_query", "text": search_info["query"]}
                    
#                     search_results = await self.conversation.google_search(search_info["query"])
#                     yield {"type": "sources", "content": search_results}
                    
#                     urls = [result["url"] for result in search_results[:5]]
#                     if urls:
#                         tables = await self._extract_tables_from_urls(urls)
#                         if tables:
#                             self.query_tables[q] = tables
#                             yield {"type": "tables", "content": tables}
        
#         # Milestone 6
#         total_queries = sum(len(queries) for queries in level2_queries.values())
#         yield {"type": "reasoning", "text": f"✅ Data collection complete: {total_queries} queries executed"}
        
#         # Milestone 7
#         yield {"type": "reasoning", "text": "🤖 Analyzing each query with AI..."}
        
#         # Phase 3: Execute send_message for each Level 2 query with tables
#         query_counter = 0
#         for branch_num, (l1_query, l2_queries) in enumerate(level2_queries.items(), 1):
#             for query_num, l2_query in enumerate(l2_queries, 1):
#                 query_counter += 1
                
#                 # Milestone 8
#                 yield {"type": "reasoning", "text": f"🔬 Analyzing query {query_counter}/{total_queries}..."}
                
#                 tables = self.query_tables.get(l2_query, [])
                
#                 if tables:
#                     tables_text = "\n\n---\n\n".join([
#                         f"**Table from {t['url']}:**\n\n{t['table']}"
#                         for t in tables[:10]
#                     ])
#                     query_with_tables = f"""{l2_query}

# ## Reference Tables:

# {tables_text}

# ## Instructions:
# **CRITICAL:** You MUST include relevant tables in your response.

# 1. Review all the tables above
# 2. **Copy the relevant tables directly into your response** (in markdown format)
# 3. For each table you include:
#    - Add a brief title/context
#    - Present the table
#    - Provide your analysis of what the data shows
   
# 4. Focus your analysis on:
#    - Key findings from the numerical data
#    - Patterns and trends in the numbers
#    - Notable statistics or metrics
#    - Comparisons across sources
#    - What the data reveals about the question

# 5. If a table is not relevant, skip it

# **FORMAT:** 
# For each relevant table, use this structure:
# - Brief context/title
# - The table itself (copy the markdown)
# - Your interpretation (2-3 sentences)

# Be analytical and data-driven. Include ALL relevant numerical data by copying the tables."""
#                 else:
#                     query_with_tables = l2_query
                
#                 result_content = ""
#                 async for chunk in self.conversation.send_message(query_with_tables, files):
#                     if chunk["type"] == "content":
#                         result_content += chunk["text"]
#                         # ✅ Yield each chunk as it streams
#                         # yield {"type": "content", "text": chunk["text"]}
                
#                 self.query_results.append({
#                     "branch": l1_query,
#                     "query": l2_query,
#                     "result": result_content,
#                     "tables_count": len(tables)
#                 })
        
#         # Milestone 9
#         yield {"type": "reasoning", "text": f"✅ Completed AI analysis for all {total_queries} queries"}
        
#         # Milestone 10
#         yield {"type": "reasoning", "text": "📝 Compiling comprehensive research report..."}
        
#         # Phase 4: Build final report from all query results (LLM already included tables)
#         final_report = f"""# Deep Research Report

# ## Research Question: {query}

# ---

# """
        
#         for i, l1_query in enumerate(level1_queries, 1):
#             final_report += f"\n## Branch {i}: {l1_query}\n\n"
            
#             branch_results = [r for r in self.query_results if r["branch"] == l1_query]
            
#             for j, result in enumerate(branch_results, 1):
#                 final_report += f"### Sub-query {i}.{j}: {result['query']}\n\n"
                
#                 # LLM already included relevant tables in its response
#                 final_report += result['result']
#                 final_report += "\n\n---\n\n"
        
#         # Milestone 11
#         yield {"type": "reasoning", "text": "🎯 Generating cross-branch synthesis..."}
        
#         # Phase 5: Generate final synthesis
#         synthesis_prompt = f"""Based on all the research conducted above in this conversation, create a comprehensive synthesis.

# Original question: "{query}"

# Review all the branch analyses above (which already include relevant tables and data) and create:

# ### 1. Executive Summary
# Synthesize the most important findings across all branches (3-4 paragraphs)

# ### 2. Cross-Branch Synthesis Tables

# Create NEW tables that consolidate findings:

# **Key Findings Comparison Table:**
# | Branch | Main Findings | Supporting Data | Implications |
# |--------|---------------|-----------------|--------------|

# **Quantitative Summary Table:**
# | Metric/Data Point | Value | Source | Significance |
# |-------------------|-------|--------|--------------|

# ### 3. Patterns & Insights
# - Cross-cutting themes
# - Contradictions or gaps
# - Emerging trends

# ### 4. Overall Implications
# What do these findings mean for the original question? What are the practical implications?

# ### 5. Future Research Needs
# What gaps exist? What should be investigated further?

# Be comprehensive and analytical. Create synthesis tables that combine data from multiple branches."""

#         synthesis_content = ""
#         async for chunk in self.conversation.send_message(synthesis_prompt, files):
#             if chunk["type"] == "content":
#                 synthesis_content += chunk["text"]
        
#         final_report += f"\n\n# Final Synthesis\n\n{synthesis_content}"
        
#         yield {"type": "report", "content": final_report}
        
#         # Milestone 12
#         yield {"type": "reasoning", "text": "📋 Generating research methodology documentation..."}
        
#         # Phase 6: Research Summary & Methodology
#         from simple_search_claude_streaming_with_web_search import ClaudeConversation
#         methodology_conversation = ClaudeConversation()
        
#         total_searches = sum(len(queries) for queries in level2_queries.values()) + len(level1_queries)
#         total_tables = sum(len(tables) for tables in self.query_tables.values())
        
#         query_breakdown = ""
#         for i, l1_query in enumerate(level1_queries, 1):
#             query_breakdown += f"\n**Branch {i}:** {l1_query}\n"
#             l2_queries = level2_queries.get(l1_query, [])
#             for j, l2_query in enumerate(l2_queries, 1):
#                 tables_count = len(self.query_tables.get(l2_query, []))
#                 query_breakdown += f"  - Sub-query {i}.{j}: {l2_query} ({tables_count} tables)\n"
        
#         methodology_prompt = f"""Research question: "{query}"

# Research structure:
# - Branches: {len(level1_queries)}
# - Total queries: {total_searches}
# - Tables analyzed: {total_tables}

# Query breakdown:
# {query_breakdown}

# Create a detailed research methodology summary with:

# 1. **Methodology Overview Table**
# 2. **Research Process Breakdown Table**
# 3. **Data Collection Summary**
# 4. **Quality Assessment**
# 5. **Step-by-step Process**
# 6. **Transparency & Limitations**

# Be detailed and academic."""

#         methodology_content = ""
#         async for chunk in methodology_conversation.send_message(methodology_prompt, files):
#             if chunk["type"] == "content":
#                 methodology_content += chunk["text"]
        
#         yield {"type": "research_summary", "content": methodology_content}
        
#         # Final milestone
#         yield {"type": "reasoning", "text": "✅ Deep research complete!"}
    
#     async def _extract_tables_from_urls(self, urls: List[str]) -> List[Dict[str, any]]:
#         """Extract tables from URLs using the scraper"""
#         try:
#             scrape_results = await scrape_tables_parallel(
#                 urls,
#                 browser_pool=self.browser_pool,
#                 timeout=60000
#             )
            
#             all_tables = []
#             for url, tables in scrape_results.items():
#                 if tables:
#                     for table in tables:
#                         all_tables.append({
#                             'url': url,
#                             'table': table
#                         })
            
#             return all_tables
        
#         except Exception as e:
#             print(f"❌ Table extraction error: {e}")
#             import traceback
#             traceback.print_exc()
#             return []
    
#     async def _generate_level1_queries(self, query: str) -> List[str]:
#         """Generate 1-2 broad sub-questions"""
        
#         prompt = f"""Given this research question: "{query}"

# Break it down into 5-6 distinct, broad sub-questions that would comprehensively cover different aspects of the topic.

# Rules:
# - Each sub-question should explore a different angle or dimension
# - Questions should be broad enough to warrant multiple detailed searches
# - Cover: definitions, current state, applications, challenges, future directions, comparisons, etc.
# - Questions should be independently answerable

# Format your response ONLY as a JSON array of strings:
# ["question 1", "question 2"]"""

#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=1000,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         response_text = response.content[0].text.strip()
        
#         try:
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             queries = json.loads(response_text)
#             return queries[:2]
#         except:
#             lines = [line.strip() for line in response_text.split('\n') if line.strip()]
#             queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
#             return queries[:2]
    
#     async def _generate_level2_queries(self, l1_query: str, original_query: str) -> List[str]:
#         """Generate 1-2 specific questions for each L1 query"""
        
#         prompt = f"""Original research question: "{original_query}"

# Sub-question to expand: "{l1_query}"

# Generate 2-3 specific, searchable queries that would help answer this sub-question in detail.

# Rules:
# - Each query should be specific and directly searchable on Google
# - Queries should dig deeper into the sub-question
# - Include current year (2025) if time-sensitive
# - Make queries independently understandable (include context)

# Format your response ONLY as a JSON array of strings:
# ["specific query 1", "specific query 2"]"""

#         response = await self.client.messages.create(
#             model=self.model,
#             max_tokens=500,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         response_text = response.content[0].text.strip()
        
#         try:
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             queries = json.loads(response_text)
#             return queries[:1]
#         except:
#             lines = [line.strip() for line in response_text.split('\n') if line.strip()]
#             queries = [line.lstrip('0123456789.-) ') for line in lines if len(line) > 10]
#             return queries[0:1]