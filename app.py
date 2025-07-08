import streamlit as st
import google.generativeai as genai
import time
import random
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import json
import re

# Configure page
st.set_page_config(
    page_title="DSA Mastery", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemma-3-27b-it')

# Data structures and topics
DS_TOPICS = {
    "Arrays": ["Basic Operations", "Two Pointers", "Sliding Window", "Prefix Sum"],
    "Strings": ["Pattern Matching", "String Manipulation", "Palindromes", "Anagrams"],
    "Linked Lists": ["Singly Linked List", "Doubly Linked List", "Circular Lists", "Merging"],
    "Stacks": ["Basic Operations", "Expression Evaluation", "Monotonic Stack", "Next Greater Element"],
    "Queues": ["Basic Operations", "Circular Queue", "Priority Queue", "Deque"],
    "Recursion": ["Base Cases", "Recursive Relations", "Tree Recursion", "Tail Recursion"],
    "Backtracking": ["N-Queens", "Sudoku", "Permutations", "Combinations", "Maze Solving"],
    "Binary Trees": ["Tree Traversal", "Tree Construction", "Tree Properties", "Path Problems"],
    "Binary Search Trees": ["BST Operations", "BST Validation", "BST to Array", "Balanced BST"],
    "Heaps": ["Min Heap", "Max Heap", "Heap Sort", "Priority Queue", "K-way Merge"],
    "Hashing": ["Hash Tables", "Hash Functions", "Collision Resolution", "Two Sum Problems"],
    "Dynamic Programming": ["1D DP", "2D DP", "Memoization", "Tabulation", "Classic Problems"],
    "Bit Manipulation": ["Basic Operations", "Bit Tricks", "XOR Properties", "Counting Bits"],
    "Graphs": ["Graph Representation", "BFS", "DFS", "Shortest Path"],
    "Greedy": ["Activity Selection", "Huffman Coding", "Fractional Knapsack", "MST"],
    "Divide & Conquer": ["Merge Sort", "Quick Sort", "Binary Search", "Closest Pair"],
    "Trie": ["Prefix Tree", "Auto-complete", "Word Search", "Longest Prefix"],
    "Segment Trees": ["Range Query", "Range Update", "Lazy Propagation", "Applications"]
}

DIFFICULTY_LEVELS = ["Newbie", "Easy", "Medium", "Hard", "Nightmare"]

# Initialize session state
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'user_solution' not in st.session_state:
    st.session_state.user_solution = ""
if 'hint_count' not in st.session_state:
    st.session_state.hint_count = 0
if 'solved_problems' not in st.session_state:
    st.session_state.solved_problems = []

def generate_problem(topic, difficulty, model):
    """Generate a coding problem using Gemini"""
    prompt = f"""
    Generate a {difficulty} level coding problem for {topic}. 
    
    Return the response in this exact JSON format:
    {{
        "title": "Problem Title",
        "description": "Detailed problem description with examples",
        "input_format": "Input format description",
        "output_format": "Output format description",
        "constraints": "Problem constraints",
        "example_input": "Sample input",
        "example_output": "Sample output",
        "hints": ["Hint 1", "Hint 2", "Hint 3"],
        "solution_approach": "High-level approach to solve",
        "time_complexity": "Expected time complexity",
        "space_complexity": "Expected space complexity"
    }}
    
    Make sure the problem is educational and focuses on understanding {topic} concepts.
    """
    
    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Failed to parse response"}
    except Exception as e:
        return {"error": f"Error generating problem: {str(e)}"}

def visualize_array(arr, title="Array Visualization"):
    """Create interactive array visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(arr))),
        y=arr,
        text=arr,
        textposition='auto',
        marker_color='lightblue',
        marker_line_color='navy',
        marker_line_width=2
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        height=400
    )
    
    return fig

def visualize_sorting_steps(arr, algorithm="bubble_sort"):
    """Visualize sorting algorithm steps"""
    steps = []
    arr_copy = arr.copy()
    
    if algorithm == "bubble_sort":
        n = len(arr_copy)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr_copy[j] > arr_copy[j + 1]:
                    arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                    steps.append(arr_copy.copy())
    
    return steps

def visualize_binary_tree(nodes, title="Binary Tree"):
    """Create binary tree visualization"""
    if not nodes:
        return go.Figure()
    
    # Calculate positions for binary tree layout
    fig = go.Figure()
    
    # Simple binary tree visualization (can be expanded)
    levels = {}
    for i, node in enumerate(nodes):
        level = int(math.log2(i + 1)) if i > 0 else 0
        if level not in levels:
            levels[level] = []
        levels[level].append((i, node))
    
    # Add nodes
    for level, level_nodes in levels.items():
        y_pos = -level
        for i, (idx, value) in enumerate(level_nodes):
            x_pos = (i - len(level_nodes)/2) * (4 - level)
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[y_pos],
                mode='markers+text',
                text=[str(value)],
                textposition='middle center',
                marker=dict(size=40, color='lightblue', line=dict(width=2, color='navy')),
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_heap(arr, heap_type="min"):
    """Visualize heap structure"""
    fig = go.Figure()
    
    # Create heap visualization
    for i in range(len(arr)):
        level = int(math.log2(i + 1)) if i > 0 else 0
        y_pos = -level
        x_pos = (i - (2**level - 1)) * (8 / (2**level))
        
        color = 'lightgreen' if heap_type == 'min' else 'lightcoral'
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos],
            mode='markers+text',
            text=[str(arr[i])],
            textposition='middle center',
            marker=dict(size=40, color=color, line=dict(width=2, color='darkgreen' if heap_type == 'min' else 'darkred')),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{heap_type.capitalize()} Heap Visualization",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_recursion_tree(func_name, depth=4):
    """Create recursion tree visualization"""
    fig = go.Figure()
    
    # Example for fibonacci recursion
    if func_name == "fibonacci":
        def add_fib_nodes(n, x=0, y=0, level=0, parent_x=None, parent_y=None):
            if level > depth or n <= 1:
                return
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[f'fib({n})'],
                textposition='middle center',
                marker=dict(size=30, color='lightblue', line=dict(width=2, color='navy')),
                showlegend=False
            ))
            
            # Add edge from parent
            if parent_x is not None:
                fig.add_trace(go.Scatter(
                    x=[parent_x, x], y=[parent_y, y],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))
            
            # Add child nodes
            offset = 2 ** (depth - level - 1)
            add_fib_nodes(n-1, x-offset, y-1, level+1, x, y)
            add_fib_nodes(n-2, x+offset, y-1, level+1, x, y)
        
        add_fib_nodes(5)
    
    fig.update_layout(
        title=f"Recursion Tree - {func_name}",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_bit_operations(num1, num2, operation):
    """Visualize bit manipulation operations"""
    fig = go.Figure()
    
    # Convert to binary representation
    bin1 = format(num1, '08b')
    bin2 = format(num2, '08b')
    
    if operation == "AND":
        result = num1 & num2
    elif operation == "OR":
        result = num1 | num2
    elif operation == "XOR":
        result = num1 ^ num2
    else:
        result = num1
    
    result_bin = format(result, '08b')
    
    # Create visualization
    positions = list(range(8))
    
    fig.add_trace(go.Bar(
        x=positions, y=[1]*8,
        text=list(bin1),
        textposition='auto',
        name=f'Number 1 ({num1})',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=positions, y=[2]*8,
        text=list(bin2),
        textposition='auto',
        name=f'Number 2 ({num2})',
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        x=positions, y=[3]*8,
        text=list(result_bin),
        textposition='auto',
        name=f'Result ({result})',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f"Bit {operation} Operation",
        xaxis_title="Bit Position",
        yaxis=dict(showticklabels=False),
        height=400
    )
    
    return fig

def visualize_dp_table(dp_table, problem_name):
    """Visualize DP table"""
    fig = go.Figure(data=go.Heatmap(
        z=dp_table,
        colorscale='Viridis',
        showscale=True,
        text=dp_table,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=f"DP Table - {problem_name}",
        height=400
    )
    
    return fig

def main():
    st.title("🧠 DSA Mastery - AI-Powered Learning")
    st.markdown("*Master Data Structures and Algorithms with AI guidance*")
    
    # Initialize Gemini
    try:
        model = init_gemini()
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        st.stop()
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🎯 Problem Selection")
        
        # Topic selection
        selected_topic = st.selectbox(
            "Choose Data Structure/Algorithm:",
            list(DS_TOPICS.keys()),
            index=0
        )
        
        # Subtopic selection
        if selected_topic:
            subtopic = st.selectbox(
                "Choose Subtopic:",
                DS_TOPICS[selected_topic],
                index=0
            )
        
        # Difficulty selection
        difficulty = st.selectbox(
            "Select Difficulty:",
            DIFFICULTY_LEVELS,
            index=1
        )
        
        # Generate problem button
        if st.button("🎲 Generate New Problem", type="primary"):
            with st.spinner("Generating problem..."):
                problem = generate_problem(f"{selected_topic} - {subtopic}", difficulty, model)
                if "error" not in problem:
                    st.session_state.current_problem = problem
                    st.session_state.start_time = time.time()
                    st.session_state.hint_count = 0
                    st.session_state.user_solution = ""
                    st.rerun()
                else:
                    st.error(problem["error"])
        
        # Stats section
        st.header("📊 Your Stats")
        st.metric("Problems Solved", len(st.session_state.solved_problems))
        if st.session_state.solved_problems:
            avg_time = sum(p.get('time_taken', 0) for p in st.session_state.solved_problems) / len(st.session_state.solved_problems)
            st.metric("Avg Time", f"{avg_time:.1f}s")
    
    # Main content area
    if st.session_state.current_problem:
        problem = st.session_state.current_problem
        
        # Timer
        if st.session_state.start_time:
            elapsed = time.time() - st.session_state.start_time
            st.metric("⏱️ Time Elapsed", f"{elapsed:.1f}s")
        
        # Problem display
        st.header(f"📝 {problem['title']}")
        
        # Problem tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Problem", "Solution", "Hints", "Visualization"])
        
        with tab1:
            st.markdown("### Problem Description")
            st.write(problem['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Format:**")
                st.code(problem['input_format'])
                st.markdown("**Example Input:**")
                st.code(problem['example_input'])
            
            with col2:
                st.markdown("**Output Format:**")
                st.code(problem['output_format'])
                st.markdown("**Example Output:**")
                st.code(problem['example_output'])
            
            st.markdown("**Constraints:**")
            st.write(problem['constraints'])
            
            # Code editor
            st.markdown("### 💻 Your Solution")
            user_code = st.text_area(
                "Write your solution here:",
                value=st.session_state.user_solution,
                height=300,
                placeholder="# Write your solution here\ndef solve():\n    pass"
            )
            st.session_state.user_solution = user_code
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🏃‍♂️ Run Code"):
                    st.info("Code execution simulation - integrate with code runner")
            
            with col2:
                if st.button("✅ Submit"):
                    if user_code.strip():
                        time_taken = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                        st.session_state.solved_problems.append({
                            'title': problem['title'],
                            'difficulty': difficulty,
                            'time_taken': time_taken,
                            'hints_used': st.session_state.hint_count
                        })
                        st.success(f"Problem submitted! Time: {time_taken:.1f}s")
                    else:
                        st.warning("Please write a solution first")
            
            with col3:
                if st.button("🔄 Reset"):
                    st.session_state.user_solution = ""
                    st.session_state.hint_count = 0
                    st.rerun()
        
        with tab2:
            st.markdown("### 🎯 Solution Approach")
            st.write(problem['solution_approach'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Time Complexity:**")
                st.code(problem['time_complexity'])
            with col2:
                st.markdown("**Space Complexity:**")
                st.code(problem['space_complexity'])
            
            # Generate detailed solution
            if st.button("🔍 Get Detailed Solution"):
                with st.spinner("Generating detailed solution..."):
                    solution_prompt = f"""
                    Provide a detailed solution for this problem:
                    {problem['title']}
                    {problem['description']}
                    
                    Include:
                    1. Step-by-step explanation
                    2. Complete code solution
                    3. Explanation of the approach
                    """
                    try:
                        response = model.generate_content(solution_prompt)
                        st.markdown("### 📚 Detailed Solution")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating solution: {str(e)}")
        
        with tab3:
            st.markdown("### 💡 Hints")
            
            for i, hint in enumerate(problem['hints']):
                if st.button(f"Reveal Hint {i+1}", key=f"hint_{i}"):
                    st.info(f"**Hint {i+1}:** {hint}")
                    st.session_state.hint_count = max(st.session_state.hint_count, i+1)
            
            # AI-powered personalized hint
            if st.button("🤖 Get AI Hint"):
                hint_prompt = f"""
                User is solving: {problem['title']}
                Their current code: {st.session_state.user_solution}
                
                Provide a helpful hint without giving away the complete solution.
                Focus on the next logical step they should take.
                """
                try:
                    response = model.generate_content(hint_prompt)
                    st.success(f"**AI Hint:** {response.text}")
                    st.session_state.hint_count += 1
                except Exception as e:
                    st.error(f"Error generating hint: {str(e)}")
        
        with tab4:
            st.markdown("### 📊 Algorithm Visualization")
            
            # Topic-specific visualizations
            if "array" in selected_topic.lower():
                sample_array = [64, 34, 25, 12, 22, 11, 90]
                fig = visualize_array(sample_array, "Sample Array")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🎬 Visualize Sorting"):
                    steps = visualize_sorting_steps(sample_array)
                    progress_bar = st.progress(0)
                    chart = st.empty()
                    
                    for i, step in enumerate(steps):
                        fig = visualize_array(step, f"Sorting Step {i+1}")
                        chart.plotly_chart(fig, use_container_width=True)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
            
            elif "binary tree" in selected_topic.lower():
                sample_tree = [1, 2, 3, 4, 5, 6, 7]
                fig = visualize_binary_tree(sample_tree, "Binary Tree Structure")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🌳 Show Tree Traversals"):
                    st.code("Inorder: 4 2 5 1 6 3 7\nPreorder: 1 2 4 5 3 6 7\nPostorder: 4 5 2 6 7 3 1")
            
            elif "bst" in selected_topic.lower() or "binary search tree" in selected_topic.lower():
                sample_bst = [4, 2, 6, 1, 3, 5, 7]
                fig = visualize_binary_tree(sample_bst, "Binary Search Tree")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("BST Property: Left < Root < Right")
            
            elif "heap" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    min_heap = [1, 3, 6, 5, 9, 8]
                    fig = visualize_heap(min_heap, "min")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    max_heap = [9, 8, 6, 5, 3, 1]
                    fig = visualize_heap(max_heap, "max")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif "recursion" in selected_topic.lower():
                fig = visualize_recursion_tree("fibonacci")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🔄 Show Call Stack"):
                    st.code("""
Call Stack Visualization:
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2) → 1
│   │   └── fib(1) → 1
│   └── fib(2) → 1
└── fib(3) → 2
                    """)
            
            elif "bit manipulation" in selected_topic.lower():
                col1, col2 = st.columns(2)
                with col1:
                    num1 = st.number_input("Number 1", value=12, min_value=0, max_value=255)
                with col2:
                    num2 = st.number_input("Number 2", value=10, min_value=0, max_value=255)
                
                operation = st.selectbox("Operation", ["AND", "OR", "XOR"])
                fig = visualize_bit_operations(num1, num2, operation)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "dynamic programming" in selected_topic.lower() or "dp" in selected_topic.lower():
                # Example DP table for fibonacci
                dp_table = [[0, 1, 1, 2, 3, 5, 8, 13]]
                fig = visualize_dp_table(dp_table, "Fibonacci DP")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("DP Table shows memoized results to avoid recomputation")
            
            elif "hashing" in selected_topic.lower():
                # Hash table visualization
                hash_data = {
                    'Key': ['apple', 'banana', 'cherry', 'date'],
                    'Hash': [hash('apple') % 10, hash('banana') % 10, hash('cherry') % 10, hash('date') % 10],
                    'Value': [5, 7, 3, 9]
                }
                
                fig = go.Figure(data=[
                    go.Bar(x=hash_data['Key'], y=hash_data['Hash'], name='Hash Values'),
                ])
                fig.update_layout(title="Hash Function Visualization", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            elif "backtracking" in selected_topic.lower():
                st.markdown("### 🔄 Backtracking Visualization")
                st.code("""
N-Queens Solution Tree:
Try Queen at (0,0) ✓
├── Try Queen at (1,2) ✓
│   ├── Try Queen at (2,1) ✗ (Conflict!)
│   └── Backtrack...
└── Try Queen at (1,3) ✓
    └── Continue...
                """)
            
            elif "graph" in selected_topic.lower():
                st.info("Graph visualization - BFS/DFS traversal animations")
                # Add graph visualization here
            
            # Interactive complexity analysis
            st.markdown("### 📈 Complexity Analysis")
            
            n_values = list(range(1, 1001, 50))
            complexities = {
                "O(1)": [1] * len(n_values),
                "O(log n)": [math.log2(n) for n in n_values],
                "O(n)": n_values,
                "O(n log n)": [n * math.log2(n) for n in n_values],
                "O(n²)": [n**2 for n in n_values],
            }
            
            complexity_df = pd.DataFrame(complexities, index=n_values)
            st.line_chart(complexity_df)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to DSA Mastery!
        
        Your AI-powered companion for mastering Data Structures and Algorithms.
        
        ### Features:
        - 🎯 **AI-Generated Problems**: Personalized problems based on your skill level
        - ⏱️ **Timed Practice**: Contest-style environment with timers
        - 💡 **Smart Hints**: AI-powered hints that guide without spoiling
        - 📊 **Interactive Visualizations**: See algorithms in action
        - 📈 **Progress Tracking**: Monitor your improvement over time
        
        ### How to Start:
        1. Select a topic from the sidebar
        2. Choose your difficulty level
        3. Click "Generate New Problem"
        4. Start coding and learning!
        
        **Ready to begin your DSA journey?** 👈 Use the sidebar to get started!
        """)
        
        # Display sample visualization
        st.markdown("### 🎨 Sample Visualization")
        sample_data = [3, 7, 1, 9, 4, 6, 8, 2, 5]
        fig = visualize_array(sample_data, "Sample Array - Click Generate Problem to start!")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    import math
    main()
